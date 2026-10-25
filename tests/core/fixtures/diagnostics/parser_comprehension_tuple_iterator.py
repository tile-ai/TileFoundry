"""A concat list comprehension over a tuple instead of a range."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="rejected", target=CpuTarget())
class ComprehensionTupleIterator:
    @func
    def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
        return tf.concat([x for index in (0,)], axis=0)
