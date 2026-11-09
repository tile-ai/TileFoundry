"""A loop over tf.tile given four arguments."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="looping", target=cpu.Target())
class TileFourArguments:
    @func
    def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
        out = tf.add(seed, seed)
        for row in tf.tile(1, 2, 3, 4):
            out = tf.add(x[row, :], seed)
        return out
