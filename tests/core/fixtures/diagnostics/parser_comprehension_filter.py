"""A concat list comprehension with a filter."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="rejected", target=CpuTarget())
class ComprehensionFilter:
    @func
    def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
        return tf.concat([x for index in range(1) if index], axis=0)
