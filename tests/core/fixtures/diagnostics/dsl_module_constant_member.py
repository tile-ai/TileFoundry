"""A Module class body with a constant member, which is none of the three member kinds."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="only", target=CpuTarget())
class _BadMember:
    @func
    def only(x: Tensor[(2, 4), "f32"], g: Tensor[(4,), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.rms_norm(x, g)

    budget = 3
