"""CUDA target codegen: the device translation unit and CUDA's side of a call."""

from __future__ import annotations

from tilefoundry.codegen.cuda import abi as _abi  # noqa: F401 -- registers CUDA's side
from tilefoundry.codegen.cuda import emit as _emit  # noqa: F401 -- emitter autodiscovery
from tilefoundry.codegen.cuda import host_launch as _host_launch  # noqa: F401
from tilefoundry.codegen.cuda.context import CudaCodegenContext
from tilefoundry.codegen.cuda.module import CudaLinkableModule

__all__ = ["CudaCodegenContext", "CudaLinkableModule"]
