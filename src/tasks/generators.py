from . import DENOISE_SCHEDULERS
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F 

import sacrebleu

from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import copy
from sacremoses import MosesDetokenizer

detokenize = MosesDetokenizer().detokenize

def mbr_duplicate(tensor, duplicate_size):
    batch_size, tail = tensor.shape[0], tensor.shape[1:]
    tensor = tensor.unsqueeze(1)
    tensor = tensor.repeat((1, duplicate_size, *([1]*len(tail))))
    return tensor.view(batch_size * duplicate_size, *tail)

def _decode(self, toks_mask):
    toks, mask = toks_mask
    toks = toks[~mask]
    # s = " ".join(list(map(str, toks.int().cpu().tolist())))
    s = self.dictionary.string(
        toks.int().cpu(),
        self.eval_bleu_remove_bpe,
        unk_string="UNKNOWNTOKENINHYP",
    )
    if self.tokenizer:
        s = self.tokenizer.decode(s)
    s = detokenize(s.split())
    return s

def _find_mbr_best_idx(candidates, lang):
    tok = "intl" if lang=="de" else "13a"
    size = len(candidates)
    return torch.tensor([
        [
            sacrebleu.corpus_bleu([candidates[x]], [[candidates[y]]], tokenize=tok).score
            for y in range(size) if x != y
        ] for x in range(size)
    ]).mean(dim=1).argmax(dim=0)       

def _find_mbr_best(self, seqs, indices):
    if self.diverse_generation == "length":
        return [
            i + _find_mbr_best_idx(seqs[i:i+self.mbr], self.lang)
            for i in range(indices[0], indices[-1]+1, self.mbr)
        ]
    elif self.diverse_generation == "sampling":
        return [
            indices[i] + self.mbr * _find_mbr_best_idx([seqs[j] for j in range(indices[i], indices[-1]+1, self.mbr)], self.lang)
            for i in range(self.length_beam)
        ]
    elif self.diverse_generation == "none":
        beam_candidate_ids = [
            j + _find_mbr_best_idx(seqs[j:j+self.mbr], self.lang)
            for j in range(indices[0], indices[-1]+1, self.mbr)
        ]
        candidate_seqs = [seqs[idx] for idx in beam_candidate_ids]
        return [beam_candidate_ids[_find_mbr_best_idx(candidate_seqs, self.lang)]]
class DiffusionGenerator(ABC):
    def __init__(self, noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer):
        self.noise_scheduler = noise_scheduler
        self.timesteps = denoise_scheduler(task_config.denoise_steps)
        self.oracle_length = task_config.oracle_length
        self.length_beam = task_config.beam
        self.mbr = task_config.mbr

        self.lang = task_config.target_lang
        self.clamp = task_config.clamp
        self.interpolate = task_config.interpolate
        self.guidance = task_config.denoise_guidance

        self.eval_bleu_remove_bpe = task_config.eval_bleu_remove_bpe
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.pad = dictionary.pad()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.tokenizer = tokenizer
        
        self.diverse_generation = task_config.diverse_generation
        # self.pool = mp.Pool(mp.cpu_count())

    @abstractmethod
    def denoise(self, models, enc_out, sample):
        # models[0] for diffusion, model[1:] for classifier guidance
        pass

    def classifier_free_guidance_denoise(self, 
        diffusion_model, encoder_out, x_t, tgt_padding_mask, t,
        prev_x_start=None, partial_mask=None, clamp=False, interpolate=False
    ):
        model_out = 0
        if (1 + self.guidance) != 0:
            model_out += diffusion_model.forward_denoise(
                encoder_out, x_t, tgt_padding_mask, t,
                prev_x_start=prev_x_start, partial_mask=partial_mask,
                clamp=self.clamp, interpolate=self.interpolate
            ) * (1 + self.guidance)
        
        if self.guidance != 0:
            model_out -= diffusion_model.forward_denoise(
                None, x_t, tgt_padding_mask, t,
                prev_x_start=prev_x_start, partial_mask=partial_mask,
                clamp=self.clamp, interpolate=self.interpolate
            ) * self.guidance
        
        # (1 + guidance) * conditional - guidance * unconditional
        return model_out
    

    def mbr_select(self, predict_tokens, padding_mask):
        if self.length_beam * self.mbr == 1:    # avoid mbr to save time
            return torch.arange(0, predict_tokens.size(0)).unsqueeze(1).to(predict_tokens)
        decode = partial(_decode, self)
        gen_seqs = list(map(decode, zip(predict_tokens, padding_mask)))
        # [_decode(self, (tokens, mask)) for tokens, mask in zip(predict_tokens, padding_mask)]
        sample_indices = [
            range(i, i+self.length_beam * self.mbr) 
            for i in range(0, len(gen_seqs), self.length_beam * self.mbr)
        ]
        find_mbr_best = partial(_find_mbr_best, self, gen_seqs)
        result_ids = list(map(find_mbr_best, sample_indices))
        # [_find_mbr_best(self, gen_seqs, indices)  for indices in sample_indices]
        return torch.tensor(result_ids).to(predict_tokens)
        
    def generate(self, models, sample):
        sample = copy.deepcopy(sample)
        model = models[0]
        assert (model.scheduler is self.noise_scheduler)
        # encode
        encoder_out = (
            model.forward_encoder(sample["net_input"]["src_tokens"])
            if ("net_input" in sample) and (self.guidance != -1) else None
        )   # None for unconditional sampling
        tgt_length = (
            model.forward_length(encoder_out, length_beam=self.length_beam)["prediction"].flatten() 
            if not self.oracle_length else None
        )  # (batch_size*length_beam, )
        # duplicate & init samples
        encoder_out = ({
            "feature": mbr_duplicate(encoder_out["feature"], self.length_beam * self.mbr),
            "padding_mask": mbr_duplicate(encoder_out["padding_mask"], self.length_beam * self.mbr)
        } if encoder_out is not None else None)
        tgt_length = mbr_duplicate(tgt_length, self.mbr) if tgt_length is not None else None
        sample["target"] = mbr_duplicate(sample["target"], self.length_beam*self.mbr)
        sample = model.init_target(sample, target_length=tgt_length)
        # print(torch.abs((tgt_length - (~(sample["target"] == self.pad())).sum(-1))).mean())
        batch_size, max_seqlen = sample["target_padding_mask"].shape
        sample["init_noise"] = torch.randn(batch_size, max_seqlen, model.latent_dim).to(model.timeW)
        # denoising
        pred_x_start = self.denoise(models, encoder_out, sample)
        prediction = model.forward_generator(pred_x_start, sample["target_padding_mask"], mask_output_padding=True)

        # postprocess
        mbr_selection_ids = self.mbr_select(prediction["prediction"], sample["target_padding_mask"])
        selections = (
            prediction["logits"][mbr_selection_ids].max(dim=-1).values,
            prediction["prediction"][mbr_selection_ids],
            sample["target_padding_mask"][mbr_selection_ids]
        )
        return [
            [
                {
                    "tokens": tokens[~pad_mask],
                    "score": pos_score.sum(-1) / (pos_score!=0).sum(-1),
                    "positional_scores": pos_score[~pad_mask],
                    "alignment": None
                } for pos_score, tokens, pad_mask in zip(pos_score_topk, tokens_topk, pad_mask_topk)
            ] for pos_score_topk, tokens_topk, pad_mask_topk in zip(*selections)
        ]
        

class OracleGenerator(DiffusionGenerator):
    def __init__(self, noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer):
        super().__init__(noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer)
    def denoise(self, models, enc_out, sample):
        return -1
    def generate(self, models, sample):
        sample = models[0].init_target(copy.deepcopy(sample), target_length=None)
        return [
            [{
                "tokens": tokens[~pad_mask],
                "score": torch.zeros_like(tokens[0]).float(),
                "positional_scores": torch.zeros_like(tokens[~pad_mask]).float(),
                "alignment": None
            }] for tokens, pad_mask in zip(sample["target"], sample["target_padding_mask"])
        ]


class DDPMGenerator(DiffusionGenerator):    # FIXME: to support guidance
    def __init__(self, noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer) -> None:
        super().__init__(noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer)
    
    def denoise(self, models, enc_out, sample):
        diffusion = models[0]
        batch_size = sample["init_noise"].size(0)
        tgt_pad_mask = sample["target_padding_mask"]
        x_cur = sample["init_noise"]
        if "partial_mask" in sample:
            partial_mask = sample["partial_mask"] 
            x_0 = diffusion.inference(sample["target"], sample["target_padding_mask"])
        else:
            partial_mask = None
        
        prev_x_start = torch.zeros_like(x_cur) if diffusion.self_conditioning else None
        
        def diffusion_denoise(x_t, t):
            nonlocal prev_x_start
            if partial_mask is not None:
                x_t[partial_mask] = x_0[partial_mask]  
            model_out = self.classifier_free_guidance_denoise(
                diffusion, enc_out, x_t, sample["target_padding_mask"], t,
                prev_x_start=prev_x_start, partial_mask=partial_mask,
                clamp=self.clamp, interpolate=self.interpolate 
            )
            pred_x_start = diffusion.predict_x_start(x_t, model_out, t)
            prev_x_start = pred_x_start if prev_x_start is not None else None 
            return model_out

        for _cur_time, _nxt_time in tqdm(zip(self.timesteps[:-1], self.timesteps[1:])):
            cur_time = torch.ones((batch_size, )).to(x_cur) * _cur_time
            nxt_time = torch.ones((batch_size, )).to(x_cur) * _nxt_time
            model_out = diffusion_denoise(x_cur, cur_time)
            pred_x_start = diffusion.predict_x_start(x_cur, model_out, cur_time)
            pred_noise = diffusion.predict_noise(x_cur, model_out, cur_time)
            
            scale_cur_t = self.noise_scheduler.scale(cur_time)
            scale_next_t = self.noise_scheduler.scale(nxt_time)
            sigma_square = (1 - scale_next_t**2) / (1 - scale_cur_t**2) * (1 - (scale_cur_t / scale_next_t)**2)
            direction = torch.sqrt(1 - scale_next_t**2 - sigma_square)

            x_cur = scale_cur_t[:, None, None] * pred_x_start.to(scale_cur_t) + \
                    direction[:, None, None] * pred_noise.to(scale_cur_t) + \
                    torch.sqrt(sigma_square)[:, None, None] * torch.randn_like(pred_noise, dtype=scale_cur_t.dtype)
            x_cur = x_cur.to(pred_noise)

        return pred_x_start


class DDIMSolverGenerator(DiffusionGenerator):
    def __init__(self, noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer) -> None:
        super().__init__(noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer)

    def denoise(self, models, enc_out, sample):
        diffusion = models[0]
        batch_size = sample["init_noise"].size(0)
        tgt_pad_mask = sample["target_padding_mask"]
        x_cur = sample["init_noise"]
        if "partial_mask" in sample:
            partial_mask = sample["partial_mask"] 
            x_0 = diffusion.inference(sample["target"], sample["target_padding_mask"])
        else:
            partial_mask = None
        
        prev_x_start = torch.zeros_like(x_cur) if diffusion.self_conditioning else None
        
        def diffusion_denoise(x_t, t):
            nonlocal prev_x_start
            if partial_mask is not None:
                x_t[partial_mask] = x_0[partial_mask]       
            model_out = self.classifier_free_guidance_denoise(
                diffusion, enc_out, x_t, sample["target_padding_mask"], t,
                prev_x_start=prev_x_start, partial_mask=partial_mask,
                clamp=self.clamp, interpolate=self.interpolate 
            )
            pred_x_start = diffusion.predict_x_start(x_t, model_out, t)
            prev_x_start = pred_x_start if prev_x_start is not None else None 
            return model_out
        for _cur_time, _nxt_time in tqdm(zip(self.timesteps[:-1], self.timesteps[1:])):
            cur_time = torch.ones((batch_size, )).to(x_cur) * _cur_time
            nxt_time = torch.ones((batch_size, )).to(x_cur) * _nxt_time
            model_out = diffusion_denoise(x_cur, cur_time)
            pred_x_start = diffusion.predict_x_start(x_cur, model_out, cur_time)
            pred_noise = diffusion.predict_noise(x_cur, model_out, cur_time)
            x_cur = diffusion.sample_x_t(pred_x_start, nxt_time, noise=pred_noise) # FIXME: try buggy
        return pred_x_start



class CEDIGenerator(DiffusionGenerator):
    def __init__(self, noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer) -> None:
        super().__init__(noise_scheduler, denoise_scheduler, task_config, dictionary, tokenizer)
        self.timestep_cur = torch.linspace(1, 1e-3, len(self.timesteps))
    
    def denoise(self, models, enc_out, sample):
        diffusion = models[0]
        batch_size = sample["init_noise"].size(0)
        tgt_pad_mask = sample["target_padding_mask"]
        x_cur = sample["init_noise"]
        if "partial_mask" in sample:
            partial_mask = sample["partial_mask"] 
            x_0 = diffusion.inference(sample["target"], sample["target_padding_mask"])
        else:
            partial_mask = None
        
        prev_x_start = torch.zeros_like(x_cur) if diffusion.self_conditioning else None
        
        def diffusion_denoise(x_t, t):
            nonlocal prev_x_start
            if partial_mask is not None:
                x_t[partial_mask] = x_0[partial_mask]       
            model_out = self.classifier_free_guidance_denoise(
                diffusion, enc_out, x_t, sample["target_padding_mask"], t,
                prev_x_start=prev_x_start, partial_mask=partial_mask,
                clamp=self.clamp, interpolate=self.interpolate 
            )
            pred_x_start = diffusion.predict_x_start(x_t, model_out, t)
            prev_x_start = pred_x_start if prev_x_start is not None else None 
            return model_out
        
        for _cur_time, _nxt_time in tqdm(zip(self.timesteps[:-1], self.timestep_cur[1:])):
            cur_time = torch.ones((batch_size, )).to(x_cur) * _cur_time
            nxt_time = torch.ones((batch_size, )).to(x_cur) * _nxt_time
            model_out = diffusion_denoise(x_cur, cur_time)
            pred_x_start = diffusion.predict_x_start(x_cur, model_out, cur_time)
            pred_noise = diffusion.predict_noise(x_cur, model_out, cur_time)
            x_cur = diffusion.sample_x_t(pred_x_start, nxt_time, noise=pred_noise) 

        return pred_x_start
