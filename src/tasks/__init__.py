from .denoise_scheduler import (
    UniformHalfLogSNRScheduler, UniformNoiseScheduler, 
    UniformTimeDenoiseScheduler, QuadraticNoiseScheduler,
    TimeWarpScheduler
)
DENOISE_SCHEDULERS = {
    "uniform_time": UniformTimeDenoiseScheduler,
    "uniform_noise": UniformNoiseScheduler,
    "quadratic_noise": QuadraticNoiseScheduler,
    "uniform_half_log_snr": UniformHalfLogSNRScheduler,
    "time_warp": TimeWarpScheduler
}

from .generators import (
    DDIMSolverGenerator, DDPMGenerator, OracleGenerator, CEDIGenerator,

)
GENERATORS = {
    "ddim": DDIMSolverGenerator,
    "ddpm": DDPMGenerator,
    "oracle": OracleGenerator,
    "cedi": CEDIGenerator
}
