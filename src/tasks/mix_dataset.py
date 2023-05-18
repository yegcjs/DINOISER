from fairseq.data import FairseqDataset, data_utils

import torch
import torch.nn.functional as F

import numpy as np

import torch.distributed as dist

from math import ceil

class MixDataset(FairseqDataset):
    def __init__(self, conditional_dataset, unconditional_dataset, uncond_ratio=0, generate_mode=False):
        super().__init__()
        self.conditional_dataset = conditional_dataset
        self.uncondoffset = (
            len(conditional_dataset) 
            if conditional_dataset is not None 
            else 0
        )
        self.unconditional_dataset = unconditional_dataset
        self.uncond_ratio = uncond_ratio

        self.generate_mode = generate_mode

        if self.generate_mode:
            assert unconditional_dataset is None

        # if unconditional_dataset is None:
        #     assert uncond_ratio == 0
    
    def __len__(self):
        if self.unconditional_dataset is None:
            return len(self.conditional_dataset) 
        elif self.conditional_dataset is None:
            return len(self.unconditional_dataset)
        else:
            return len(self.conditional_dataset) + len(self.unconditional_dataset)

    def __getitem__(self, index):
        if index >= self.uncondoffset:
            sample = self.unconditional_dataset[index - self.uncondoffset]
            sample["id"] += self.uncondoffset
        else:
            sample = self.conditional_dataset[index]
        return sample

    def filter_indices_by_size(self, indices, max_sizes):
        if self.conditional_dataset is None:
            return self.unconditional_dataset.filter_indices_by_size(indices, max_sizes)
        if self.unconditional_dataset is None:
            return self.conditional_dataset.filter_indices_by_size(indices, max_sizes)

        cond_indices = indices[indices < self.uncondoffset]
        uncond_indices = indices[indices >= self.uncondoffset] - self.uncondoffset

        cond_indices, cond_ignore = self.conditional_dataset.filter_indices_by_size(cond_indices, max_sizes)
        uncond_indices, uncond_ignore = self.unconditional_dataset.filter_indices_by_size(uncond_indices, max_sizes)
        uncond_indices, uncond_ignore = uncond_indices + self.uncondoffset, [idx+self.uncondoffset for idx in uncond_ignore]
        return np.concatenate([cond_indices, uncond_indices]), ([cond_ignore] + [uncond_ignore])

    def size(self, index):
        if index >= self.uncondoffset:
            sample = self.unconditional_dataset[index - self.uncondoffset]
        else:
            sample = self.conditional_dataset[index]
        return (sample["source"].size(0), sample["target"].size(0))

    def collater(self, samples):
        if self.generate_mode:
            return self.conditional_dataset.collater(samples)
        if self.conditional_dataset is None:
            uncond_samples = self.unconditional_dataset.collater(samples)
            del uncond_samples["net_input"]
            return {
                "conditional": {},
                "unconditional": uncond_samples
            }
        if self.unconditional_dataset is None:
            return {
                "conditional": self.conditional_dataset.collater(samples), 
                "unconditional": {}
            }
        
        cond_samples = self.conditional_dataset.collater(
            [item for item in samples if (item["id"] < self.uncondoffset)]
        )
        uncond_samples = self.unconditional_dataset.collater(
            [item for item in samples if (item["id"] >= self.uncondoffset)]
        )
        del uncond_samples["net_input"]
        return {
            "conditional": cond_samples,
            "unconditional": uncond_samples
        }

    def ordered_indices(self):
        if self.conditional_dataset is None:
            return self.unconditional_dataset.ordered_indices()
        elif self.unconditional_dataset is None:
            return self.conditional_dataset.ordered_indices()
        else:
            return np.concatenate([
                self.conditional_dataset.ordered_indices(), 
                self.unconditional_dataset.ordered_indices() + self.uncondoffset
            ])

    def _rebatch(self, batches):
        if not dist.is_initialized():
            return batches
        world_size = dist.get_world_size()
        num_split = world_size - ((len(batches) - 1) % world_size)
        first_batch = batches[0]
        batch_size = ceil(first_batch.size / num_split)
        split_first_batch = [
            first_batch[i: i+batch_size]
            for i in range(0, first_batch.size, batch_size)
        ]
        batches = split_first_batch + batches[1:]
        
        assert len(split_first_batch) == num_split
        assert len(batches) % world_size == 0
        
        return batches

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        if self.conditional_dataset is None:
            return self._rebatch(self.unconditional_dataset.batch_by_size(
                indices, max_tokens=max_tokens, max_sentences=max_sentences, 
                required_batch_size_multiple=required_batch_size_multiple
            ))
        if self.unconditional_dataset is None:
            return self._rebatch(self.conditional_dataset.batch_by_size(
                indices, max_tokens=max_tokens, max_sentences=max_sentences, 
                required_batch_size_multiple=required_batch_size_multiple
            ))

        assert max_tokens is not None
        max_tokens_uncond =  int(max_tokens * self.uncond_ratio)
        max_tokens_cond = max_tokens - max_tokens_uncond
        
        cond_indices = indices[indices < self.uncondoffset]
        uncond_indices = indices[indices >= self.uncondoffset] - self.uncondoffset

        cond_batches = self._rebatch(self.conditional_dataset.batch_by_size(
            cond_indices, max_tokens=max_tokens_cond, max_sentences=max_sentences, 
            required_batch_size_multiple=required_batch_size_multiple
        ))
        uncond_batches = self._rebatch(self.unconditional_dataset.batch_by_size(
            uncond_indices, max_tokens=max_tokens_uncond, max_sentences=max_sentences, 
            required_batch_size_multiple=required_batch_size_multiple
        ))

        uncond_batches = [batch + self.uncondoffset for batch in uncond_batches]

        if len(cond_batches) <= len(uncond_batches):
            return [
                np.concatenate([cond_batches[offset], uncond_batches[base+offset]])
                for base in range(0, len(uncond_batches)-len(cond_batches)+1, len(cond_batches))
                for offset in range(len(cond_batches))
            ]
        else:
            raise NotImplementedError