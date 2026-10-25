from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_N_MAX = 1023
_NESTED_N = DimVar("nested_n", 1, _N_MAX)
_DISPATCH_BOUND = 128


@module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
class _NestedDispatch:
    """A dispatch variant whose body calls a second dispatch prototype."""

    @func
    def inner(x: Tensor[(_NESTED_N,), "f32"]) -> Tensor[(_NESTED_N,), "f32"]:
        pass

    @inner.specialize(RangePattern("nested_n", 1, _DISPATCH_BOUND - 1))
    def inner_small(x: Tensor[(_NESTED_N,), "f32"]) -> Tensor[(_NESTED_N,), "f32"]:
        return tf.add(x, x)

    @inner.specialize(RangePattern("nested_n", _DISPATCH_BOUND, _N_MAX))
    def inner_big(x: Tensor[(_NESTED_N,), "f32"]) -> Tensor[(_NESTED_N,), "f32"]:
        return tf.add(tf.add(x, x), x)

    @func
    def mid(x: Tensor[(_NESTED_N,), "f32"]) -> Tensor[(_NESTED_N,), "f32"]:
        pass

    @mid.specialize(RangePattern("nested_n", 1, _DISPATCH_BOUND - 1))
    def mid_small(x: Tensor[(_NESTED_N,), "f32"]) -> Tensor[(_NESTED_N,), "f32"]:
        return inner(x)  # noqa: F821

    @mid.specialize(RangePattern("nested_n", _DISPATCH_BOUND, _N_MAX))
    def mid_big(x: Tensor[(_NESTED_N,), "f32"]) -> Tensor[(_NESTED_N,), "f32"]:
        return inner(x)  # noqa: F821

    @func
    def run(x: Tensor[(_NESTED_N,), "f32"]) -> Tensor[(_NESTED_N,), "f32"]:
        return mid(x)  # noqa: F821
