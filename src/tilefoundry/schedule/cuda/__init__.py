from tilefoundry.ir.target import CudaTarget

from ..registry import register_schedule_backend
from .backend import CudaCtaBackend

register_schedule_backend(CudaTarget, level="cta", backend=CudaCtaBackend())

__all__ = ["CudaCtaBackend"]
