"""A concat given its inputs as separate positional arguments."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="direct", target=CpuTarget())
class ConcatDirectInputs:
    @func
    def direct(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.concat(a, b, axis=0)
