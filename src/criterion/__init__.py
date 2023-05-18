from .time_samplers import (
    UniformSampler, LossAwareSampler, TimeWarpSampler, AnnelDecaySampler, AnnelAscendSampler, ClippedSampler, SmoothedClippedSampler
)

TIME_SAMPLERS = {
    "uniform": UniformSampler,
    "lossaware": LossAwareSampler,
    "time_warp": TimeWarpSampler,
    "anneal_decay": AnnelDecaySampler,
    'anneal_ascend': AnnelAscendSampler,
    "clipped": ClippedSampler,
    "clipped_s": SmoothedClippedSampler
}

from .diffusion_loss import DiffusionCLMLoss, DiffusionPostEditLoss
from .length_loss import LengthClassificationLoss