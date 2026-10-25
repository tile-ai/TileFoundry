"""A concat list comprehension with two generators."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="rejected", target=CpuTarget())
class ComprehensionTwoGenerators:
    @func
    def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.concat([x for first in range(1) for second in range(2)], axis=0)
