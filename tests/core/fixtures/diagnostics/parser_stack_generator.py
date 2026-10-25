"""A stack over a generator expression."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="generator", target=CpuTarget())
class StackGenerator:
    @func
    def generator(x: Tensor[(2, 4), "f32"]) -> Tensor[(2, 4), "f32"]:
        return tf.stack((x[index] for index in range(2)), axis=0)
