"""CUDA target and compiler registrations for TileFoundry."""

from tilefoundry_nvidia.cuda.target import CudaArchitecture, CudaDevice, CudaTarget

__all__ = ["CudaArchitecture", "CudaDevice", "CudaTarget"]

from tilefoundry_nvidia.cuda import codegen, ir, passes, visitor_registry  # noqa: E402,F401
