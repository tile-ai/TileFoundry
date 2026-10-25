"""A ``pass`` prototype whose variants square ``[1, 4)`` and double ``[4, 8)``."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_S = DimVar("S", 1, 7)


@module(entry="main", target=CudaTarget("nvidia.h200_sxm"))
class Dispatch:
    @func
    def main(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
        pass

    @main.specialize(RangePattern("S", 1, 3))
    def small_shape(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
        return tf.mul(x, x)

    @main.specialize(RangePattern("S", 4, 7))
    def large_shape(x: Tensor[(_S,), "f32"]) -> Tensor[(_S,), "f32"]:
        return tf.add(x, x)  # noqa: F821
