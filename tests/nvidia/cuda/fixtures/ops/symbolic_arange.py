"""An arange whose extent is a ceildiv of a runtime dimension."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_N = DimVar("arange_n", 1, 16)


@module(entry="_symbolic_arange", target=CudaTarget("nvidia.h200_sxm"))
class SymbolicArange:
    @func
    def _symbolic_arange(x: Tensor[(_N,), "f32"]):
        return tf.arange(Tensor[(ceildiv(_N - 1, 2),), "i32"], start=1, step=2)
