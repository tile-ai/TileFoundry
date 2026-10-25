from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="rt", target=CpuTarget())
class CastF4e2m1:
    @func
    def rt(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f4e2m1"]:
        return tf.cast(x, "f4e2m1")
