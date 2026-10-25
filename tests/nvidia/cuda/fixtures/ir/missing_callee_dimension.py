from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_CALL_M = DimVar("call_m", 1, 16)
_N_MAX = 1023
_CALL_N = DimVar("call_n", 1, _N_MAX)
_DISPATCH_BOUND = 128


@module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
class _MissingCalleeDimension:
    """A caller dimension independent of the dimension its callee dispatches on."""

    @func
    def pick(x: Tensor[(_CALL_N,), "f32"]) -> Tensor[(_CALL_N,), "f32"]:
        pass

    @pick.specialize(RangePattern("call_n", 1, _DISPATCH_BOUND - 1))
    def pick_small(x: Tensor[(_CALL_N,), "f32"]) -> Tensor[(_CALL_N,), "f32"]:
        return tf.add(x, x)

    @pick.specialize(RangePattern("call_n", _DISPATCH_BOUND, _N_MAX))
    def pick_big(x: Tensor[(_CALL_N,), "f32"]) -> Tensor[(_CALL_N,), "f32"]:
        return tf.add(tf.add(x, x), x)

    @func
    def run(
        x: Tensor[(_CALL_M,), "f32"], k: Tensor[(_CALL_N,), "f32"]
    ) -> Tensor[(_CALL_N,), "f32"]:
        return pick(k)  # noqa: F821
