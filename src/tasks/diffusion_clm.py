import torch

from typing import Optional

from dataclasses import dataclass, field
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationTask, 
    TranslationConfig, 
    load_langpair_dataset,
    EVAL_BLEU_ORDER,
)
import sacrebleu

from . import GENERATORS, DENOISE_SCHEDULERS

from .mix_dataset import MixDataset

@dataclass
class DiffusionCLMConfig(TranslationConfig):
    # decoding
    oracle_length: bool = field(
        default=False,
        metadata={
            "help": "whether to decode with oracle length"
        }
    )
    interpolate: bool = field(
        default=False,
        metadata={
            "help": "interpolate model out"
        }
    )
    beam: int = field(
        default=1,
        metadata={
            "help": "length beam"
        }
    )
    mbr: int = field(
        default=1,
        metadata={
            "help": "number of mbr candidates"
        }
    )
    diverse_generation: str = field(
        default="none",
        metadata={
            "help": "generate diverse outputs"
        }
    )
    solver: str = field(
        default="ddim",
        metadata={
            "help": "sample solver"
        }
    )
    denoise_scheduler: str = field(
        default="uniform_time",
        metadata={
            "help": "denoise scheduler"
        }
    )
    denoise_steps: int = field(
        default=200,
        metadata={
            "help": "number of decoding steps"
        }
    )
    denoise_start: float = field(
        default=1.,
        metadata={
            "help": "where to start denosing"
        }
    )
    denoise_end: float = field(
        default=1e-3,
        metadata={
            "help": "last denoising timestep"
        }
    )
    denoise_guidance: float = field(
        default=0,
        metadata={
            "help": "w for guidance denoising: (1+w) f(x, c) - w f(x)"
        }
    )
    
    # training
    unconditional_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "auxiliary data for unconditional training"
        }
    )
    unconditional_ratio: Optional[float] = field(
        default=0,
        metadata={
            "help": "randomly select part of data for unconditional training"
        }
    )

    # generate
    generate_mode: bool = field(
        default=False,
        metadata={
            "help": "doing fairseq generation"
        }
    )

    # tricks
    clamp: bool = field(
        default=False,
        metadata={
            "help": "whether to apply clamp"
        }
    )
    # override
    left_pad_source: bool = field(
        default=False, metadata={"help": "pad the source on the left"}
    )


@register_task("diffusion_clm", dataclass=DiffusionCLMConfig)
class DiffusionCLM(TranslationTask):
    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]
        conditional_dataset = (
            load_langpair_dataset(
                data_path,
                split,
                self.cfg.source_lang,
                self.src_dict,
                self.cfg.target_lang,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=False,  # XXX
                left_pad_target=False,  # XXX
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
                prepend_bos=True
            )  if (
                self.cfg.unconditional_ratio != 1 or
                split != self.cfg.train_subset 
            )
            else None 
        )
        unconditional_dataset = (
            load_langpair_dataset(  # TODO, FIXME: change it to mololingual dataset
                self.cfg.unconditional_data,
                split,
                self.cfg.source_lang,
                self.src_dict,
                self.cfg.target_lang,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=False,  # XXX
                left_pad_target=False,  # XXX
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
                prepend_bos=True
            )
            if (
                (self.cfg.unconditional_data is not None) and 
                (self.cfg.unconditional_ratio != 0) and 
                (split == self.cfg.train_subset)
            )
            else None
        )
        self.datasets[split] = MixDataset(
            conditional_dataset, unconditional_dataset, 
            uncond_ratio=self.cfg.unconditional_ratio,
            generate_mode=self.cfg.generate_mode,
        )

    def build_generator(self, models, args, **kwargs):
        diffusion_model = models[0]
        denoise_scheduler = DENOISE_SCHEDULERS[self.cfg.denoise_scheduler](
            diffusion_model.scheduler, self.cfg.denoise_end, self.cfg.denoise_start
        )
        return GENERATORS[self.cfg.solver](
            diffusion_model.scheduler, denoise_scheduler, self.cfg, self.tgt_dict, self.tokenizer,
        )

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad):
        criterion.set_update(update_num)
        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample, eval=True)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):
        return super()._inference_with_bleu(generator, sample["conditional"], model)

    @torch.no_grad()
    def inference_step(self, generator, models, sample, **unused):
        gen_out = generator.generate(models, sample)
        return gen_out