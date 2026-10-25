"""A weight converter that squares the weight it is given."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
class _Doubled:
    @func
    def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.add(x, w)

    @run.converter("w")
    def _convert_w(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.mul(w, w)
