"""A concat list comprehension that unpacks into a tuple target."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="rejected", target=CpuTarget())
class ComprehensionTupleTarget:
    @func
    def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
        return tf.concat([x for first, second in range(1)], axis=0)
