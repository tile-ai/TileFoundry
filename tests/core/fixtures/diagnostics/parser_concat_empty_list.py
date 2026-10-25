"""A concat over an empty input list."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="empty", target=CpuTarget())
class ConcatEmptyList:
    @func
    def empty(x: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
        return tf.concat([], axis=0)
