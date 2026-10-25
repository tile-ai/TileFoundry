"""A concat given its axis as a second positional argument."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="positional_attribute", target=CpuTarget())
class ConcatPositionalAxis:
    @func
    def positional_attribute(
        a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]
    ) -> Tensor[(2, 4), "f32"]:
        return tf.concat([a, b], 0)
