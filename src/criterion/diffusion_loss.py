import torch
import torch.nn.functional as F 

from dataclasses import dataclass, field
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq import metrics
from fairseq.utils import new_arange

from .utils import (
    masked_mean_flat, reduce_logging_outputs
)

from . import TIME_SAMPLERS
from ..models.diffusion import ModelOutputType

@dataclass
class DiffusionCLMLossConfig(FairseqDataclass):
    time_sampler: ChoiceEnum(TIME_SAMPLERS.keys()) = field(
        default="uniform",
        metadata={
            "help": "time sampler"
        }
    )
    timesteps: int = field(
        default=-1,
        metadata={
            "help": "number timesteps applied during training, -1 for continuous"
        }
    )
    decay_factor: float = field(
        default=5e-6,
        metadata={
            "help": "sampling decay factor"
        }
    )
    min_t: float = field(
        default=1e-3
    )
    max_t: float = field(
        default=1.
    )
    importance_sampling: bool = field(
        default=False,
        metadata={
            "help": "whether to apply importance sampling to rescale the loss"
        }
    )
    diffusion_loss_type: str = field(
        default="l2-x0",   # (l1, l2, huber)-(noise, x0, velocity, mle, reweighted) | ce
        metadata={
            "help": "XXX"
        }
    )
    ignore_partial_masked_loss: bool = field(
        default=False
    )


@register_criterion("diffusion_clm_loss", dataclass=DiffusionCLMLossConfig)
class DiffusionCLMLoss(FairseqCriterion):
    def __init__(self, task, cfg):
        super().__init__(task)
        self.timesampler = TIME_SAMPLERS[cfg.time_sampler](
            cfg.timesteps, cfg.min_t, cfg.max_t, cfg.importance_sampling, cfg.decay_factor
        )
        loss_type = cfg.diffusion_loss_type.split('-')
        self.loss_func = loss_type[0]   # l1, l2, huber
        self.loss_type = loss_type[1] if len(loss_type)>1 else None # None, noise, x0, reweighted, mll, velocity
        self.ignore_partial_masked_loss = cfg.ignore_partial_masked_loss
        self.update_num = 0

    def compute_loss(self, model, model_out, x_t, x_0, noise, t, sample, ignore_mask=None):
        if self.loss_func == "ce":
            pred_x_0 = model.predict_x_start(x_t, model_out, t)
            generator_logits = model.generator(pred_x_0, sample["target_padding_mask"])["logits"]
            diffusion_loss = masked_mean_flat(
                F.cross_entropy(generator_logits, sample["target"], reduction="none"), 
                ignore_mask
            )
            nll = torch.zeros_like(diffusion_loss)
        else:
            if self.loss_type in ("noise", "mle", "reweighted", "trunc_snr"):
                prediction = model.predict_noise(x_t, model_out, t)
                target = noise # model.predict_noise(x_t.detach(), x_0.detach(), t, model_out_type=ModelOutputType.X0)
                if self.loss_type == "noise":
                    weight = torch.ones_like(t)
                elif self.loss_type == "mle":
                    weight = (model.scheduler.diffusion_coeff(t) / model.scheduler.std(t))**2
                elif self.loss_type == "reweighted":
                    weight = model.scheduler.diffusion_coeff(t) ** 2
                elif self.loss_type == "trunc_snr":
                    weight = torch.exp((-model.scheduler.log_snr(t)).clamp(0, 1e12))
            elif self.loss_type.startswith("P2"):
                _, k, gamma = self.loss_type.split('_')
                k, gamma = float(k), float(gamma)
                # FIXME: assume x0 
                prediction = model.predict_x_start(x_t, model_out, t)
                target = x_0 
                snr = (model.scheduler.scale(t) / model.scheduler.std(t))**2
                weight = 1 / (k + snr)**gamma
            else:
                if self.loss_type == "x0":
                    prediction = model.predict_x_start(x_t, model_out, t)
                    target = x_0
                elif self.loss_type == "velocity":
                    prediction = model.predict_velocity(x_t, model_out, t)
                    target = model.predict_velocity(x_t, x_0, t, model_out_type=ModelOutputType.X0)
                elif self.loss_type == "mix_x0_eps":
                    prediction = model_out
                    target = torch.cat([x_0, model.predict_noise(x_t, x_0, t, ModelOutputType.X0)], dim=-1)
                    assert prediction.shape == target.shape
                else:
                    raise NotImplementedError
                weight = torch.ones_like(t)
            
            diffusion_loss = {
                "l1": F.l1_loss,
                "l2": F.mse_loss,
                "huber": F.smooth_l1_loss
            }[self.loss_func](prediction, target, reduction="none")
            # tT
            tT_loss = masked_mean_flat(model.sample_x_t(x_0, torch.ones_like(t), noise=torch.zeros_like(x_0))**2, ignore_mask) # * weight
            diffusion_loss = masked_mean_flat(diffusion_loss, ignore_mask) * weight + tT_loss
            batch_size, seqlen = x_0.size(0), x_0.size(1)
            generator_logits = model.forward_generator(x_0, sample["target_padding_mask"])["logits"]
            nll = masked_mean_flat(
                F.cross_entropy(
                    generator_logits.view(batch_size*seqlen, -1), sample["target"].flatten(), 
                    reduction="none"
                ).view(batch_size, seqlen), 
                ignore_mask
            )
        return diffusion_loss, nll


    def _forward(self, model, sample, eval=False):
        batch_size = 0 if sample == {} else sample["target"].size(0)
        if batch_size == 0:
            logging_output = {
                "nsentences": 0,
                "ntokens": 0,
                "loss": 0,
                "diffusion_loss": 0,
                "nll": 0,
            }
            return 0, 1, logging_output
        
        sample = model.init_target(sample)
        
        encoder_out = (
            model.forward_encoder(sample["net_input"]["src_tokens"])
            if "net_input" in sample else None
        )
        x0 = model.forward_inference(sample["target"], sample["target_padding_mask"])["latent"]
        t, weight = self.timesampler(x0, update_num=self.update_num)

        prev_x_start = (torch.zeros_like(x0) if model.self_conditioning else None)
        partial_mask = getattr(sample, "partial_mask", None)    # always None for non-partial diffusion, keep this to support parital diffusion setting
        
        ignore_mask = sample["target_padding_mask"]
        if (partial_mask is not None) and (not self.ignore_partial_masked_loss):
            ignore_mask = ignore_mask | partial_mask

        noise = torch.randn_like(x0)
        x_t = model.sample_x_t(x0, t, noise=noise, partial_mask=partial_mask)

        model_out = model.forward_denoise(
            encoder_out, x_t, sample["target_padding_mask"], t,
            prev_x_start=prev_x_start, partial_mask=partial_mask
        )
        diffusion_loss, nll = self.compute_loss(
            model, model_out, x_t, x0, noise, t, sample, ignore_mask=ignore_mask
        )

        if model.self_conditioning:
            with torch.no_grad():
                prev_x_start = model.predict_x_start(x_t.detach(), model_out.detach(), t)
            model_out = model.forward_denoise(
                encoder_out, x_t, sample["target_padding_mask"], t,
                prev_x_start=prev_x_start, partial_mask=partial_mask
            )
            diffusion_loss_sc, nll_sc = self.compute_loss(
                model, model_out, x_t, x0, noise, t, sample, ignore_mask=ignore_mask
            )
            diffusion_loss += diffusion_loss_sc
            nll += nll_sc
        
        if not eval:
            self.timesampler.update_sampler(t, diffusion_loss.detach(), model)

        diffusion_loss = (diffusion_loss * weight).mean()
        nll = (nll * weight).mean()
        loss = diffusion_loss + nll
        logging_output = {
            "nsentences": batch_size,
            "ntokens": (~sample["target_padding_mask"]).sum(),
            "loss": loss.data,
            "diffusion_loss": diffusion_loss.data,
            "nll": nll.data,
        }
        return loss, 1, logging_output
    
    def set_update(self, update_num):
        self.update_num = update_num

    def forward(self, model, sample, eval=False, reduce=True):

        # if self.unconditional_ratio <= 0:
        #     return self._forward(model, sample, eval=eval)
        
        # split samples
        # batch_size = sample["target"].size(0)
        # uncond_flag = torch.rand((batch_size, )) < self.unconditional_ratio

        # uncond_sample = {
        #     "target": sample["target"][uncond_flag]
        # }
        # cond_sample = {
        #     "net_input": {"src_tokens": sample["net_input"]["src_tokens"][~uncond_flag]},
        #     "target": sample["target"][~uncond_flag]
        # }
        
        # new 
        cond_sample, uncond_sample = sample["conditional"], sample["unconditional"]

        # compute loss separately
        uncond_loss, _, uncond_loggingout = self._forward(model, uncond_sample, eval=eval)
        cond_loss, _, cond_loggingout = self._forward(model, cond_sample, eval=eval)

        uncond_tokens_cnt, cond_tokens_cnt = uncond_loggingout["ntokens"], cond_loggingout["ntokens"]
        total_tokens = uncond_tokens_cnt + cond_tokens_cnt
        
        def weighted_mean(uncond_x, cond_x):
            if total_tokens == 0:
                return torch.tensor(0., requires_grad=True)
            return (uncond_x * uncond_tokens_cnt + cond_x * cond_tokens_cnt) / total_tokens

        loss = weighted_mean(uncond_loss, cond_loss)
        diffusion_loss = weighted_mean(uncond_loggingout["diffusion_loss"], cond_loggingout["diffusion_loss"])
        nll = weighted_mean(uncond_loggingout["nll"], cond_loggingout["nll"])
        
        logging_output = {
            "nsentences": cond_loggingout["nsentences"] + uncond_loggingout["nsentences"],
            "ntokens": total_tokens,
            "loss": loss.data,
            "diffusion_loss": diffusion_loss.data,
            "nll": nll.data
        }
        return loss, 1, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs):        
        diffusion_loss = reduce_logging_outputs(logging_outputs, "diffusion_loss", "ntokens")
        nll = reduce_logging_outputs(logging_outputs, "nll", "ntokens")
        loss = diffusion_loss + nll
        metrics.log_scalar("loss", loss)
        metrics.log_scalar("diffusion_loss", diffusion_loss)
        metrics.log_scalar("nll", nll)


@register_criterion("diffusion_postedit_loss", dataclass=DiffusionCLMLossConfig)
class DiffusionPostEditLoss(DiffusionCLMLoss):
    def __init__(self, task, cfg):
        super().__init__(task, cfg)

    def forward(self, model, sample, eval=False, reduce=True):
        # compute loss separately
        sample = model.init_target(sample)
        
        encoder_out = (
            model.forward_encoder(sample["net_input"]["src_tokens"])
            if "net_input" in sample else None
        )
        x0 = model.forward_inference(sample["target"], sample["target_padding_mask"])["latent"]
        t, weight = self.timesampler(x0, update_num=self.update_num)

        prev_x_start = None #  model.forward_inference(sample["dist_target"], sample["target_padding_mask"])["latent"]
        partial_mask = getattr(sample, "partial_mask", None)    # always None for non-partial diffusion, keep this to support parital diffusion setting
        
        ignore_mask = sample["target_padding_mask"]
        if (partial_mask is not None) and (not self.ignore_partial_masked_loss):
            ignore_mask = ignore_mask | partial_mask

        noise = torch.randn_like(x0)
        x_t = model.sample_x_t(x0, t, noise=noise, partial_mask=partial_mask)
        # print(sample["target"].shape, x_t.shape, sample["ori_target"].shape, prev_x_start.shape)
        model_out = model.forward_denoise(
            encoder_out, x_t, sample["target_padding_mask"], t,
            prev_x_start=prev_x_start, partial_mask=partial_mask
        )
        diffusion_loss, nll = self.compute_loss(
            model, model_out, x_t, x0, noise, t, sample, ignore_mask=ignore_mask
        )
        
        if not eval:
            self.timesampler.update_sampler(t, diffusion_loss.detach(), model)

        diffusion_loss = (diffusion_loss * weight).mean()
        nll = (nll * weight).mean()
        loss = diffusion_loss + nll
        logging_output = {
            "nsentences": x_t.size(0),
            "ntokens": (~sample["target_padding_mask"]).sum(),
            "loss": loss.data,
            "diffusion_loss": diffusion_loss.data,
            "nll": nll.data,
        }
        return loss, 1, logging_output
