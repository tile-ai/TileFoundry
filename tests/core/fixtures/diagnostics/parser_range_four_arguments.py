"""A loop over range given four arguments."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="looping", target=CpuTarget())
class RangeFourArguments:
    @func
    def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
        out = tf.add(seed, seed)
        for row in range(1, 2, 3, 4):
            out = tf.add(x[row, :], seed)
        return out
