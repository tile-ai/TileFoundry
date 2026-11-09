"""A loop over tf.tile given only an extent."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="looping", target=cpu.Target())
class TileOneArgument:
    @func
    def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
        out = tf.add(seed, seed)
        for row in tf.tile(10):
            out = tf.add(x[row, :], seed)
        return out
