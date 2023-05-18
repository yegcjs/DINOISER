from abc import ABC, abstractmethod
from math import log
import torch


class SchedulerBase(ABC):
    def __init__(self):
        pass

    # @abstractmethod
    def diffusion_coeff(self, t):
        pass

    def noise_derivative(self, t):
        std = self.std(t)
        return 2 * std / (1 - std**2)**2

    # @abstractmethod
    def std(self, t):
        return ( 1 - self.scale(t)**2 )**0.5

    # @abstractmethod
    def scale(self, t):
        return ( 1 - self.std(t)**2 )**0.5

    # @abstractmethod
    def rev_std(self, std_t):
        raise NotImplementedError

    # @abstractmethod
    def log_snr(self, t):
        return 2 * torch.log(self.scale(t) / self.std(t))
    
    # @abstractmethod
    def rev_log_snr(self, log_snr_t):
        raise NotImplementedError


class LinearScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
        self.beta_min = 0.1
        self.beta_max = 20
    
    def diffusion_coeff(self, t):   # g(t)=sqrt(beta(t)), f(t)x = -1/2 * beta(t) * x
        return (self.beta_min + t * (self.beta_max - self.beta_min))**0.5

    def integral_beta_t(self, t):
        t = t
        return 0.5 * (self.beta_max - self.beta_min) * (t**2) + self.beta_min * t
        
    def scale(self, t):
        return torch.exp(-0.5 * self.integral_beta_t(t))
 
    def std(self, t):
        return ( 1 - torch.exp(-self.integral_beta_t(t)) )**0.5

    def rev_std(self, std_t):
        std_t = std_t
        b = self.beta_min
        _4ac = - 2 * (self.beta_max - self.beta_min) * torch.log(1 - std_t**2)
        return (-b + (b**2 - _4ac)) / (self.beta_max - self.beta_min)

    def log_snr(self, t):
        integral = self.integral_beta_t(t)
        return -integral - torch.log(1 - torch.exp(-integral))
    
    def rev_log_snr(self, log_snr_t):
        log_snr_t = log_snr_t
        tmp = torch.log(torch.exp(-2 * log_snr_t) + 1) 
        return 2 * tmp / (
            torch.sqrt(self.beta_min**2 + 2 * (self.beta_max - self.beta_min) * tmp) + self.beta_min
        )

class QuadraticScheduler(LinearScheduler):
    def __init__(self):
        super().__init__()
        self.beta_min = 0.1
        self.beta_max = 20
        self.sqrt_beta_min = self.beta_min**(1/2)
        self.sqrt_beta_max = self.beta_max**(1/2)
    
    def diffusion_coeff(self, t):
        return (self.sqrt_beta_min + t * (self.sqrt_beta_max - self.sqrt_beta_min))

    def integral_beta_t(self, t):
        inte = 1/3 * ((self.sqrt_beta_max - self.sqrt_beta_min)**2) * (t**3)
        inte += self.sqrt_beta_min * (self.sqrt_beta_max - self.sqrt_beta_min) * (t**2)
        inte += self.beta_min * t
        return inte

    def rev_std(self, std_t, dtype=torch.dtype):
        raise NotImplementedError

    def rev_log_snr(self, log_snr_t):
        raise NotImplementedError

class GeometricScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
        self.std_min = 1e-3
        self.std_max = 1
    
    def std(self, t):
        return self.std_min * (self.std_max / self.std_min)**t
    
    def scale(self, t):
        return (1 - self.std(t)**2)**0.5

    def rev_std(self, std_t):
        return (torch.log(std_t) - log(self.std_min)) / (log(self.std_max) - log(self.std_min))

    def log_snr(self, t):
        return torch.log(1 - self.std(t)**2 + 1e-5) - 2 * log(self.std_min) - 2 * t * (log(self.std_max) - log(self.std_min))
    
    def rev_log_snr(self, log_snr_t):
        base = log(self.std_max) - log(self.std_min)
        return (-log(self.std_min) - 0.5 * torch.log(torch.exp(log_snr_t) + 1)) / base


class SqrtScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
    
    def scale(self, t):
        return (1 - t**0.5)**0.5
    
    def std(self, t):
        return t**0.25
    
    def rev_std(self, std_t):
        return std_t**4

    def log_snr(self, t):
        return torch.log(1 - t**0.5 + 1e-5) - 0.5 * torch.log(t + 1e-5)
    
    def rev_log_snr(self, log_snr_t):
        return (1 + torch.exp(log_snr_t))**(-2)

class InterchangableScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
    
    def scale(self, t):
        return (1 - t**2)**0.5
    
    def std(self, t):
        return t
    
    def rev_std(self, std_t):
        return std_t

    def log_snr(self, t):
        return torch.log(1 - t**2 + 1e-5) - 2 * torch.log(t+1e-5)
    
    def rev_log_snr(self, log_snr_t):
        return (1 + torch.exp(log_snr_t)) ** (-0.5)

    def diffusion_coeff(self, t):
        return (2 * t / (1 - t**2)) ** 0.5