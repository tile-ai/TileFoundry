"""A concat list with a starred element."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="starred", target=CpuTarget())
class ConcatStarred:
    @func
    def starred(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.concat([a, *[b]], axis=0)
