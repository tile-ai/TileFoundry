"""A dispatch variant whose body calls a child that reads a constant."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.shapes.scaled_modules import EVALUATOR_N as _N_EVAL
from tests.nvidia.cuda.fixtures.shapes.scaled_modules import DynamicScaledChild
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="dispatch", target=CudaTarget("nvidia.h200_sxm"))
class _Dispatch:
    scaled = DynamicScaledChild

    @func
    def dispatch(x: Tensor[(_N_EVAL,), "f32"]) -> Tensor[(_N_EVAL,), "f32"]:
        pass

    @dispatch.specialize(RangePattern("N_eval", 1, 7))
    def dynamic_variant(x: Tensor[(_N_EVAL,), "f32"]) -> Tensor[(_N_EVAL,), "f32"]:
        return scaled(x)  # noqa: F821
