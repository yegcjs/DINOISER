import torch

class UniformTimeDenoiseScheduler:
    def __init__(self, noise_scheduler, end=0.001, start=1) -> None:
        self.noise_scheduler = noise_scheduler
        self.end = end
        self.start = start
    
    def __call__(self, num_denoise_steps):
        return torch.linspace(self.start, self.end, num_denoise_steps+1)


class UniformNoiseScheduler(UniformTimeDenoiseScheduler):
    def __init__(self, noise_scheduler, end=0.001, start=1) -> None:
        super().__init__(noise_scheduler, end, start)
    
    def __call__(self, num_denoise_steps):
        std = torch.linspace(self.start, self.end, num_denoise_steps+1)
        return self.noise_scheduler.rev_std(std)

class QuadraticNoiseScheduler(UniformTimeDenoiseScheduler):
    def __init__(self, noise_scheduler, end=0.001, start=1) -> None:
        super().__init__(noise_scheduler, end, start)
    
    def __call__(self, num_denoise_steps):
        std = torch.linspace(self.start**0.5, self.end**0.5, num_denoise_steps+1)**2
        return self.noise_scheduler.rev_std(std)

class UniformHalfLogSNRScheduler(UniformTimeDenoiseScheduler):
    def __init__(self, noise_scheduler, end=0.001, start=1) -> None:
        super().__init__(noise_scheduler, end, start)
    
    def __call__(self, num_denoise_steps):
        log_snr = torch.linspace(
            self.noise_scheduler.log_snr(torch.tensor(self.start)), 
            self.noise_scheduler.log_snr(torch.tensor(self.end)), 
            num_denoise_steps+1
        )
        return self.noise_scheduler.rev_log_snr(log_snr)

class TimeWarpScheduler(UniformNoiseScheduler):
    def __init__(self, noise_scheduler, time_warp, end=0.001, start=1) -> None:
        super().__init__(noise_scheduler, end, start)
        self.time_warp = time_warp
    
    def __call__(self, num_denoise_steps, end=0.001, start=1):
        raise NotImplementedError   # TODO