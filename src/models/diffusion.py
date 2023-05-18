import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import enum
from strenum import LowercaseStrEnum

from fairseq.models import BaseFairseqModel, register_model
from fairseq.dataclass import ChoiceEnum

from .vae import VAE, VQVAE

from .transformer import (
    TransformerDecoder, TransformerEncoderModel
)

from .length_predictors import (
    LengthPredictor, LengthClassifier, LengthRegressor 
)

import math

from . import NOISE_SCHEDULERS

class ModelOutputType(LowercaseStrEnum):
    """
    Which type of output the model predicts.
    """

    X0 = enum.auto()  # the model predicts x_0
    NOISE = enum.auto()  # the model predicts epsilon
    MIX_X0_EPS = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()
    # START_X_MEAN = enum.auto()


class DiffusionBottleNeck(nn.Module):
    def __init__(self, args, transformer_config) -> None:
        super().__init__()
        hidden_dim = transformer_config["hidden_dim"]
        self.time_conditional_layernorm = transformer_config["time_conditional_layernorm"]
        input_dim = args.latent_dim
        ffn_dim = hidden_dim # transformer_config["ffn_dim"] # XXX: 
        if args.self_conditioning:
            input_dim += args.latent_dim
            ffn_dim += hidden_dim
        # if not self.time_conditional_layernorm: 
        #     input_dim += 2 * args.half_time_embedding_dim
        #     ffn_dim += hidden_dim
        if args.partial_diffusion:
            input_dim += args.latent_dim
            ffn_dim += hidden_dim
            self.partial_embedding = nn.Embedding(2, args.latent_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(2 * args.half_time_embedding_dim, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_dim)
        )
        self.up_sampler = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.Tanh(),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.down_sampler = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.Tanh(),
            nn.Linear(
                ffn_dim, 
                args.latent_dim * 2 if args.model_output_type==ModelOutputType.MIX_X0_EPS
                else args.latent_dim
            )
        )
        self.dropout = nn.Dropout(args.dropout)

    def upsample(self, latent, prev_x_start=None, t_embed=None, partial_mask=None):  # t \in (eps, 1]
        # latent = self.dropout(latent)   # only dropout on latent?
        partial_embedding = ( 
            None if partial_mask is  None
            else self.partial_embedding(partial_mask)
        )
        # if not self.time_conditional_layernorm:
        #     batch_size, seqlen, t_embed_dim = latent.size(0), latent.size(1), t_embed.size(-1)
        #     t_embed = t_embed.unsqueeze(1).expand(batch_size, seqlen, t_embed_dim)
        #     concat_list = [latent, prev_x_start, t_embed, partial_embedding]
        # else:
        concat_list = [latent, prev_x_start, partial_embedding]
        while None in concat_list:
            concat_list.remove(None)
        x = torch.cat(concat_list, dim=-1)
        # x = self.dropout(x)    # XXX: important
        # return self.dropout(self.up_sampler(x))
        
        # concat_list = [latent, prev_x_start, partial_embedding]
        # while None in concat_list:
        #     concat_list.remove(None)
        # assert (len(concat_list)==1)
        # x = torch.cat(concat_list, dim=-1)
        # return self.dropout(self.up_sampler(x) + self.time_proj(t_embed).unsqueeze(1))
        seqlen = latent.size(1)
        return self.dropout(
            self.up_sampler(x) + self.time_proj(t_embed).unsqueeze(1).expand(-1, seqlen, -1)
        )
    def downsample(self, x):
        return self.down_sampler(x)


@register_model("diffusion_clm_model")
class DiffusionCLMModel(BaseFairseqModel):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__()
        # dict
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        # scheduler
        self.scheduler = NOISE_SCHEDULERS[args.scheduler]()
        # vae
        vae_config = json.loads(args.vae_config) 
        self.vae = {"vae": VAE, "vqvae": VQVAE}[args.vae_type](args, vae_config, tgt_dict)
        # timestep
        self.timeW = nn.Parameter(torch.randn(args.half_time_embedding_dim) * 16, requires_grad=False)  # 30? 16?
        # main diffusion model
        model_config = json.loads(args.diffusion_model_config)
        self.encoder = {"transformer": TransformerEncoderModel}[args.diffusion_model_type](args, model_config, src_dict)
        self.decoder = {"transformer": TransformerDecoder}[args.diffusion_model_type](args, model_config)
        # bottleneck
        self.bottleneck = DiffusionBottleNeck(args, model_config)
        # length
        self.length_predictor = {
            "none": LengthPredictor,
            "classification": LengthClassifier,
            "regression": LengthRegressor
        }[args.length_model_type](args, model_config["hidden_dim"])
        # args.
        self.length_predict_type = args.length_predict_type
        self.model_output_type = args.model_output_type
        self.latent_dim = args.latent_dim
        self.self_conditioning = args.self_conditioning
        self.partial_diffusion = args.partial_diffusion

    @staticmethod
    def add_args(parser):
        parser.add_argument("--scheduler", type=str)
        parser.add_argument("--vae-type", type=ChoiceEnum(["vae", "vqvae"]))
        parser.add_argument(
            "--vae-config", type=str
            # for vae: dim, std-type, inf-num-layers, gem-num-layers
            # for vq vae: code-size
        )
        parser.add_argument("--latent-dim", type=int)
        parser.add_argument("--half-time-embedding-dim", type=int)
        parser.add_argument(
            "--diffusion-model-type", type=str,
            help="only transformer supported for now"
        )
        parser.add_argument("--diffusion-model-config", type=str)
        parser.add_argument("--model-output-type", type=str)
        parser.add_argument("--partial-diffusion", action="store_true")
        parser.add_argument("--self-conditioning", action="store_true")

        parser.add_argument(
            "--length-predict-type", type=str # ChoiceEnum(["fixed", "absolute", "difference"])
        )
        parser.add_argument(
            "--length-model-type", type=str # ChoiceEnum(["none", "classification", "regression"])
        )
        parser.add_argument(
            "--dropout", type=float
        )

    @classmethod
    def build_model(cls, args, task):
        return cls(args, task.src_dict, task.tgt_dict)
    
    def init_target(self, sample, target_length=None):
        device = sample["target"].device
        if self.length_predict_type == "fixed":
            ori_target = sample["target"]
            pad_length = self.length_predictor.max_length - ori_target.shape[-1]
            sample["target"] = F.pad(ori_target, (0, pad_length), value=self.tgt_dict.pad())
            sample["target_padding_mask"] = torch.zeros_like(sample["target"]).bool()
        else:
            if target_length is None:   # oracle length
                sample["target_padding_mask"] = (sample["target"] == self.tgt_dict.pad())
            else:
                max_seqlen = target_length.max()
                assert max_seqlen > 0
                indices = torch.arange(max_seqlen, device=device)
                indices = indices.expand(target_length.size(0), max_seqlen)
                sample["target_padding_mask"] = (indices >= target_length[:, None])

        return sample

    def timestep_embedding(self, timesteps):
        t_proj = timesteps[:, None] * self.timeW[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

    def forward_encoder(self, input_ids, padding_mask=None):
        return self.encoder(input_ids, padding_mask=padding_mask)

    def forward_length(self, encoder_out, length_beam):
        return self.length_predictor(encoder_out, length_beam=length_beam)

    def forward_inference(self, input_tokens, padding_mask):
        # return {"latent": self.vae.embedding(input_tokens)}
        return self.vae.inference(input_tokens, padding_mask)

    def forward_generator(self, x_start, padding_mask, mask_output_padding=False):
        # logits = F.linear(x_start, self.vae.embedding.weight)
        # generator_out = {
        #     "logits": logits,
        #     "prediction": logits.argmax(dim=-1)
        # }
        generator_out = self.vae.generator(x_start, padding_mask)
        if mask_output_padding:
            generator_out["prediction"].masked_fill_(padding_mask, self.tgt_dict.pad())
            generator_out["logits"].masked_fill_(padding_mask.unsqueeze(-1), 0)
        return generator_out

    def forward_denoise(self, encoder_out, x_t, padding_mask, t, prev_x_start=None, 
                        partial_mask=None, clamp=False, interpolate=False):   # 
        t_embed = self.timestep_embedding(t)
        x = self.bottleneck.upsample(
            x_t, prev_x_start=prev_x_start, t_embed=t_embed, partial_mask=partial_mask
        )
        x = self.decoder(x, padding_mask, t_embed, encoder_out=encoder_out) # encoder_out==None for unconditional prediction

        x = self.bottleneck.downsample(x)
        
        if interpolate or clamp:
            x_start = self.predict_x_start(x_t, x, t)
            if interpolate:
                x_start = self.vae.interpolate(x_start, padding_mask)
            if clamp:
                x_start = self.vae.clamp(x_start, padding_mask) 
            x = {
                ModelOutputType.X0: self.predict_x_start,
                ModelOutputType.NOISE: self.predict_noise,
                ModelOutputType.SCORE: self.predict_score,
                ModelOutputType.VELOCITY: self.predict_velocity
            }[self.model_output_type](x_t, x_start, t, model_out_type=ModelOutputType.X0)

        return x

    def predict_x_start(self, x_t, model_out, t, model_out_type=None):
        if model_out_type is None:
            model_out_type = self.model_output_type
       
        if model_out_type in ModelOutputType.X0:
            x_0 = model_out
        else:
            scale_t, std_t = self.scheduler.scale(t), self.scheduler.std(t) # torch.double
            scale_t, std_t = scale_t[:, None, None], std_t[:, None, None]
            x_t_double, model_out_double = x_t.to(torch.double), model_out.to(torch.double)
            if model_out_type == ModelOutputType.NOISE:
                x_0 = (x_t_double - std_t * model_out_double) / scale_t
            elif model_out_type == ModelOutputType.MIX_X0_EPS:
                dim = x_t.shape[-1]
                predict_x_0 = model_out_double[:, :, :dim]
                predict_eps = model_out_double[:, :, dim:]
                x_0 = (std_t**2) * predict_x_0 + scale_t * (x_t_double - std_t * predict_eps)
            elif model_out_type == ModelOutputType.SCORE:
                x_0 = (x_t_double + (std_t**2) * model_out_double) / scale_t
            elif model_out_type == ModelOutputType.VELOCITY:
                x_0 = scale_t * x_t_double - std_t * model_out_double
            else:
                raise not NotImplementedError
        return x_0.to(x_t)

    def predict_noise(self, x_t, model_out, t, model_out_type=None):
        if model_out_type is None:
            model_out_type = self.model_output_type

        if model_out_type == ModelOutputType.NOISE:
            noise = model_out
        else:
            scale_t, std_t = self.scheduler.scale(t), self.scheduler.std(t) # torch.double
            scale_t, std_t = scale_t[:, None, None], std_t[:, None, None]
            x_t_double, model_out_double = x_t.to(torch.double), model_out.to(torch.double)
            if model_out_type == ModelOutputType.X0:
                noise = (x_t_double - scale_t * model_out_double) / std_t
            elif model_out_type == ModelOutputType.MIX_X0_EPS:
                dim = x_t.shape[-1]
                predict_x_0 = model_out_double[:, :, :dim]
                predict_eps = model_out_double[:, :, dim:]
                noise = std_t * x_t_double - scale_t * (std_t * predict_x_0 - scale_t * predict_eps)
            elif model_out_type == ModelOutputType.SCORE:
                noise = - model_out_double * std_t
            elif model_out_type == ModelOutputType.VELOCITY:
                noise = std_t * x_t_double + scale_t * model_out_double
            else:
                raise not NotImplementedError
        return noise.to(x_t)
    
    def predict_score(self, x_t, model_out, t, model_out_type=None):
        if model_out_type is None:
            model_out_type = self.model_output_type 

        if model_out_type == ModelOutputType.SCORE:
            score = model_out
        else:
            scale_t, std_t = self.scheduler.scale(t), self.scheduler.std(t) # torch.double
            scale_t, std_t = scale_t[:, None, None], std_t[:, None, None]
            x_t_double, model_out_double = x_t.to(torch.double), model_out.to(torch.double) 
            if model_out_type == ModelOutputType.X0:
                score = (scale_t * model_out_double - x_t_double) / (std_t**2)
            elif model_out_type == ModelOutputType.NOISE:
                score = - model_out_double / std_t
            elif model_out_type == ModelOutputType.MIX_X0_EPS:
                dim = x_t.shape[-1]
                predict_x_0 = model_out_double[:, :, :dim]
                predict_eps = model_out_double[:, :, dim:]
                score = - x_t_double + scale_t * predict_x_0 + (scale_t**2/std_t) * predict_eps
            elif model_out_type == ModelOutputType.VELOCITY:
                score = -x_t_double - scale_t / std_t * model_out_double
            else:
                raise NotImplementedError
        return score.to(x_t)
    
    def predict_velocity(self, x_t, model_out, t, model_out_type=None):
        if model_out_type is None:
            model_out_type = self.model_output_type 
        
        if model_out_type == ModelOutputType.VELOCITY:
            velocity = model_out
        else:
            scale_t, std_t = self.scheduler.scale(t), self.scheduler.std(t) # torch.double
            scale_t, std_t = scale_t[:, None, None], std_t[:, None, None]
            x_t_double, model_out_double = x_t.to(torch.double), model_out.to(torch.double) 
            if model_out_type == ModelOutputType.X0:
                velocity = (scale_t * x_t_double - model_out) / std_t
            elif model_out_type == ModelOutputType.NOISE:
                velocity = (model_out - std_t * x_t_double) / scale_t
            elif model_out_type == ModelOutputType.MIX_X0_EPS:
                dim = x_t.shape[-1]
                predict_x_0 = model_out_double[:, :, :dim]
                predict_eps = model_out_double[:, :, dim:]
                velocity = scale_t**3 * predict_eps - std_t**3 * predict_x_0 + scale_t * std_t * (std_t * predict_eps - scale_t * predict_x_0)
            elif model_out_type == ModelOutputType.SCORE:
                velocity = - std_t * (model_out + x_t_double) / scale_t 
            else:
                raise not NotImplementedError
        return velocity.to(x_t)
        
    def sample_x_t(self, x_0, t, noise=None, partial_mask=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        scale_t, std_t = self.scheduler.scale(t), self.scheduler.std(t) # torch.double
        x_t = scale_t[:, None, None] * x_0 + std_t[:, None, None] * noise
        if partial_mask is not None:
            x_t[partial_mask] = x_0[partial_mask]
        return x_t.to(x_0)