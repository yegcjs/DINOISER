from .noise_schedulers import (
    LinearScheduler, QuadraticScheduler, GeometricScheduler, SqrtScheduler, InterchangableScheduler
)

NOISE_SCHEDULERS = {
    "linear": LinearScheduler,
    "quadratic": QuadraticScheduler,
    "geometric": GeometricScheduler,
    "sqrt": SqrtScheduler,
    "interchangable": InterchangableScheduler
}
