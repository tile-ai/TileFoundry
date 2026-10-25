"""A concat given a tuple name instead of a list."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="named", target=CpuTarget())
class ConcatSequenceName:
    @func
    def named(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        parts = (a, b)
        return tf.concat(parts, axis=0)
