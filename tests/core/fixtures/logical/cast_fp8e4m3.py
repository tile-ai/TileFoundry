from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="rt", target=CpuTarget())
class CastFp8e4m3:
    @func
    def rt(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.cast(tf.cast(x, "fp8e4m3"), "bf16")
