"""The CUDA targets a GPU test runs its kernel on.

The sm_89 case carries the ``sm89`` marker, so it runs only on a local compute
capability 8.9 device.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry.ir.core.module import Module
from tilefoundry_cuda import CudaTarget

CUDA_TARGETS = (
    pytest.param(CudaTarget("nvidia.h200_sxm"), id="sm_90"),
    pytest.param(CudaTarget("nvidia.rtx_6000_ada"), id="sm_89", marks=pytest.mark.sm89),
)


def retarget(module: Module, target: CudaTarget) -> Module:
    """*module* with itself and each of its CUDA functions on *target*.

    A launch names its callee, so the host entry reaches the retargeted device function.
    """
    return replace(
        module,
        target=target,
        functions=tuple(
            replace(function, target=target)
            if isinstance(function.target, CudaTarget)
            else function
            for function in module.functions
        ),
    )
