"""A child that reads a constant, called on every trip of a loop."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.shapes.scaled_modules import ScaledChild
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="looped", target=CudaTarget("nvidia.h200_sxm"))
class _Looped:
    scaled = ScaledChild

    @func
    def looped(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        acc = x
        for _ in range(3):
            acc = scaled(acc)  # noqa: F821
        return acc
