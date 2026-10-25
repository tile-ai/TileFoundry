"""A weight converter that calls a child Module staged before it."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.shapes.scaled_modules import ScaledChild
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(entry="run", target=CudaTarget("nvidia.h200_sxm"))
class _Converted:
    scaled = ScaledChild

    @func
    def run(x: Tensor[(4,), "f32"], w: ConstTensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return tf.add(x, w)

    @run.converter("w")
    def _convert_w(w: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        return scaled(w)  # noqa: F821
