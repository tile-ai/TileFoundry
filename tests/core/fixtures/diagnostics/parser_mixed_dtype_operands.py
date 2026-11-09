"""A bf16 tensor added to an f32 tensor."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="mixed", target=cpu.Target())
class MixedDtypeOperands:
    @func
    def mixed(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "f32"]:
        return x + tf.cast(x, "f32")
