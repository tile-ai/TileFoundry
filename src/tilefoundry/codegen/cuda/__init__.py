"""CUDA target codegen: the device translation unit and CUDA's side of a call."""

from __future__ import annotations

from tilefoundry.codegen.cuda import abi as _abi  # noqa: F401 -- registers CUDA's side
from tilefoundry.codegen.cuda import emit as _emit  # noqa: F401 -- emitter autodiscovery
from tilefoundry.codegen.cuda.context import CudaCodegenContext

__all__ = ["CudaCodegenContext"]
