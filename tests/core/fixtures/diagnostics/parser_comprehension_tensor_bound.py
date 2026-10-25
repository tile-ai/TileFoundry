"""A concat list comprehension over a range bounded by a tensor."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="rejected", target=CpuTarget())
class ComprehensionTensorBound:
    @func
    def rejected(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
        return tf.concat([x for index in range(x)], axis=0)
