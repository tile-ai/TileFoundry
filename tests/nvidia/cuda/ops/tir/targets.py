"""The CUDA targets a GPU test runs its kernel on.

The sm_89 case carries the ``sm89`` marker, so it runs only on a local compute
capability 8.9 device.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry.ir import core
from tilefoundry_nvidia import cuda

CUDA_TARGETS = (
    pytest.param(cuda.Target("nvidia.h200_sxm"), id="sm_90"),
    pytest.param(cuda.Target("nvidia.rtx_6000_ada"), id="sm_89", marks=pytest.mark.sm89),
)


def retarget(module: core.Module, target: cuda.Target) -> core.Module:
    """*module* with itself and each of its CUDA functions on *target*.

    A launch names its callee, so the host entry reaches the retargeted device function.
    """
    return replace(
        module,
        target=target,
        functions=tuple(
            replace(function, target=target)
            if isinstance(function.target, cuda.Target)
            else function
            for function in module.functions
        ),
    )
