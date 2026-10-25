from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="rt", target=CpuTarget())
class CastF8e8m0:
    @func
    def rt(x: Tensor[(6,), "f32"]) -> Tensor[(6,), "f32"]:
        return tf.cast(tf.cast(x, "f8e8m0"), "f32")
