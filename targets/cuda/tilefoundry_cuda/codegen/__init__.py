"""CUDA target codegen: the device translation unit and CUDA's side of a call."""

from __future__ import annotations

from tilefoundry_cuda.codegen import abi as _abi  # noqa: F401 -- registers CUDA's side
from tilefoundry_cuda.codegen import emit as _emit  # noqa: F401 -- emitter autodiscovery
from tilefoundry_cuda.codegen import host_launch as _host_launch  # noqa: F401
from tilefoundry_cuda.codegen.context import CudaCodegenContext
from tilefoundry_cuda.codegen.module import CudaLinkableModule

__all__ = ["CudaCodegenContext", "CudaLinkableModule"]
