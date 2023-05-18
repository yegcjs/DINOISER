import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from torch.distributed import ReduceOp
from functools import partial
from torch.distributions.multinomial import Multinomial

import math

class TimeSampler(nn.Module):
    def __init__(self, num_timesteps, min_t, max_t, importance_sampling, decay_factor) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.min_t, self.max_t = min_t, max_t
        self.importance_sampling = importance_sampling
        self.decay_factor = decay_factor

    def scale_t(self, t):
        assert (t >= 0).all()
        assert (t <= 1).all()
        return self.min_t + (self.max_t - self.min_t) * t
    
    def unscale_t(self, t):
        return ((t - self.min_t) / (self.max_t - self.min_t)) # .clamp(0, 1)

    def update_sampler(self, ts, losses, model):
        pass

    def forward(self, x0):
        """
        input: x0
        output: t and weight
        """
        pass

class UniformSampler(TimeSampler):
    def __init__(self, num_timesteps, min_t, max_t, importance_sampling, decay_factor) -> None:
        super().__init__(num_timesteps, min_t, max_t, importance_sampling, decay_factor)
        self.step_size = 1 / num_timesteps
    
    def forward(self, x0, **unused):
        batch_size = x0.size(0)
        if self.num_timesteps == -1:
            t = torch.rand((batch_size, )).to(x0)
        else:
            t = (1 + torch.randint(0, self.num_timesteps, (batch_size, )).to(x0)) * self.step_size
        return self.scale_t(t), torch.ones_like(t)


class AnnelDecaySampler(UniformSampler):
    def __init__(self, num_timesteps, min_t, max_t, importance_sampling, decay_factor) -> None:
        super().__init__(num_timesteps, min_t, max_t, importance_sampling, decay_factor)
        self.step_size = 1 / num_timesteps
    
    def forward(self, x0, update_num):
        batch_size = x0.size(0)
        if self.num_timesteps == -1:
            t = torch.rand((batch_size, )).to(x0)
        else:
            t = (1 + torch.randint(0, self.num_timesteps, (batch_size, )).to(x0)) * self.step_size

        current_min_t = max(self.min_t, (1 - update_num * self.decay_factor)**0.5)
        scaled_t = current_min_t + (self.max_t - current_min_t) * t


        return scaled_t, torch.ones_like(t)

class AnnelAscendSampler(UniformSampler):
    def __init__(self, num_timesteps, min_t, max_t, importance_sampling, decay_factor) -> None:
        super().__init__(num_timesteps, min_t, max_t, importance_sampling, decay_factor)
        self.step_size = 1 / num_timesteps
    
    def forward(self, x0, update_num):
        batch_size = x0.size(0)
        if self.num_timesteps == -1:
            t = torch.rand((batch_size, )).to(x0)
        else:
            t = (1 + torch.randint(0, self.num_timesteps, (batch_size, )).to(x0)) * self.step_size

        current_min_t = min(self.max_t, max(self.min_t, (update_num * self.decay_factor)**0.25))
        scaled_t = current_min_t + (self.max_t - current_min_t) * t


        return scaled_t, torch.ones_like(t)

class ClippedSampler(UniformSampler):
    def __init__(self, num_timesteps, min_t, max_t, importance_sampling, decay_factor) -> None:
        super().__init__(num_timesteps, min_t, max_t, importance_sampling, decay_factor)
        self.min_t = min_t

    def update_sampler(self, ts, losses, model):
        with torch.no_grad():
            embedding = model.vae.get_embedding_weight()
            norm = (embedding ** 2).sum(-1)
            dist = norm[:, None] + norm[None, :] - 2 * torch.mm(embedding, embedding.T)
            dist.masked_fill_(torch.eye(dist.size(0), device=dist.device, dtype=torch.bool), 1e10)
            var = dist.min(dim=-1).values.mean() / embedding.size(-1) 
            self.min_t = model.scheduler.rev_std(((1 / (1 / var + 1)) ** 0.5)).item() # ((1 / (1 / var + 1)) ** 0.5).item()

class SmoothedClippedSampler(ClippedSampler):
    def update_sampler(self, ts, losses, model):
        with torch.no_grad():
            embedding = model.vae.get_embedding_weight()
            norm = (embedding ** 2).sum(-1)
            dist = norm[:, None] + norm[None, :] - 2 * torch.mm(embedding, embedding.T)
            dist.masked_fill_(torch.eye(dist.size(0), device=dist.device, dtype=torch.bool), 1e10)
            var = dist.min(dim=-1).values.mean() / embedding.size(-1) + 1
            self.min_t = model.scheduler.rev_std(((1 / (1 / var + 1)) ** 0.5)).item() # ((1 / (1 / var + 1)) ** 0.5).item() 

class LossAwareSampler(TimeSampler):
    def __init__(self, num_timesteps, min_t, max_t, importance_sampling) -> None:
        super().__init__(num_timesteps, min_t, max_t, importance_sampling)
        self.history_per_term = 10
        if num_timesteps == -1: # num timestepss
            num_timesteps = 100
        self.bin_size = 1 / num_timesteps
        self.register_buffer("_loss_history", torch.zeros(num_timesteps, 10))
        self.register_buffer("_weights", torch.zeros(num_timesteps))
        self.register_buffer("_loss_counts", torch.zeros([num_timesteps], dtype=torch.int))
    
    def update_sampler(self, local_ts, local_losses, model):
        local_ts = self.unscale_t(local_ts)
        local_ts = (local_ts / self.bin_size).long()
        if not dist.is_initialized():
            self.update_with_all_losses(local_ts, local_losses)
        else:
            batch_sizes = [
                torch.tensor([0], dtype=torch.int32, device=local_ts.device)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(
                batch_sizes,
                torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
            )

            # Pad all_gather batches to be the maximum batch size.
            batch_sizes = [x.item() for x in batch_sizes]
            max_bs = max(batch_sizes)
            local_ts = F.pad(local_ts, (0, max_bs-local_ts.size(0)), value=-1)
            local_losses = F.pad(local_losses, (0, max_bs-local_losses.size(0)), value=0)

            timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
            loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
            dist.all_gather(timestep_batches, local_ts)
            dist.all_gather(loss_batches, local_losses)
            timesteps = [
                x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
            ]
            losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
            self.update_with_all_losses(timesteps, losses)
    
    def update_with_all_losses(self, bin, losses):
        for t, loss in zip(bin, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:].clone()
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1
        if self._loss_counts[t] == self.history_per_term:
            weights = torch.sqrt(torch.mean(self._loss_history ** 2, dim=-1))
            weights *= 1 - 1e-4
            weights += 1e-4 / len(weights)
            self._weight = weights
    
    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

    def forward(self, x0):
        batch_size = x0.size(0)
        if not self._warmed_up():
            if self.num_timesteps == -1:
                t = torch.rand((batch_size, )).to(x0)
            else:
                t = 1 + torch.randint(0, self.num_timesteps).to(x0) * self.bin_size
            return self.scale_t(t), torch.ones_like(t)
        # first select bin, then uniform sample in bin
        bin = torch.multinomial(self._weight, batch_size, replacement=True)
        
        if self.importance_sampling:
            p = (self._weights[bin] / self._weights.sum()).to(x0)
            weights = 1 / (self._weights.size(0) * p)
        else:
            weights = torch.ones((batch_size, )).to(x0)
        
        if self.num_timesteps == -1:
            bin = bin.to(x0)
            t = torch.rand_like(bin) * self.bin_size + bin.to(x0) * self.bin_size
        else:
            t = (bin.to(x0) + 1) * self.bin_size
        return self.scale_t(t), weights

class TimeWarpSampler(TimeSampler):
    def __init__(self, num_timesteps, min_t, max_t, importance_sampling) -> None:
        super().__init__(num_timesteps, min_t, max_t, importance_sampling)
        self.num_bins = 100 # TODO: dynamic bin numbers
        self.register_buffer("_step_count", torch.tensor(0)) 
        self.t_bins = nn.Parameter(torch.zeros((self.num_bins, )))     # n bins
        self.u_bins = nn.Parameter(torch.zeros((self.num_bins, )))
        self.optimizer = torch.optim.AdamW(self.parameters())
    
    def update_sampler(self, ts, losses, model):
        if not self._warmed_up():
            return
        ts = self.unscale_t(ts)
        w_t = F.softmax(self.t_bins, dim=-1)
        w_t = (w_t + 1e-4) / (1 + 1e-4 * self.num_bins)
        t_sum = torch.cumsum(w_t, dim=-1)
        t_indices = (t_sum[None, :] < ts[:, None]).sum(-1).clamp(0, t_sum.size(-1)-1)
        loss_increments = torch.exp(self.u_bins + 1e-4)
        loss_bins = torch.cumsum(loss_increments, dim=-1)
        expected_loss = loss_bins[t_indices] - \
            loss_increments[t_indices] * ((t_sum[t_indices] - ts)/w_t[t_indices])
        loss_func = F.mse_loss(losses, expected_loss)
        loss_func.backward()
        self.optimizer.step()
        
        # print("t_bins", self.t_bins)
        # print("u_bins", self.u_bins)

        if dist.is_initialized():
            world_size = float(dist.get_world_size())
            for p in self.parameters():
                dist.all_reduce(p, op=ReduceOp.SUM)
                p.data /=  world_size
    
    def _warmed_up(self):
        return (self._step_count >= 100).all()

    @torch.no_grad()
    def forward(self, x0):
        if not self._warmed_up():
            self._step_count += 1
            t = torch.rand((x0.size(0), )).to(x0)
            weight = torch.ones_like(t)
            return self.scale_t(t), weight

        w_t = F.softmax(self.t_bins, dim=-1)
        w_t = (w_t + 1e-4) / (1 + 1e-4 * self.num_bins)
        t_sum = torch.cumsum(w_t, dim=-1)

        w_u = F.softmax(self.u_bins + 1e-4, dim=-1)
        cdf = torch.cumsum(w_u, dim=-1)
        
        s = torch.rand((x0.size(0), )).to(cdf)
        indices = ((cdf[None, :] < s[:, None]).sum(-1)).clamp(0, s.size(-1)-1)
        
        t = t_sum[indices].to(x0)
        if self.num_timesteps == -1:
            t = t - torch.rand_like(t) * w_t[indices]

        if self.importance_sampling:
            p = (w_t / w_t)[indices]
            weights = 1 / p.to(x0)
        else:
            weights = torch.ones_like(t)
        return self.scale_t(t), weights


# TODO: apply MCMC/Reject Sampling for the following Sampler

# g^2(t)
class DiffusionWeightedSampler(TimeSampler):
    pass
        

# g^2(t) / sigma^2
class MLLWeightedSampler(TimeSampler):
    pass

# X0 weighted
class X0WeightedSampler(TimeSampler):
    pass

# velocity weighted
class VelocityWeightedSampler(TimeSampler):
    pass

