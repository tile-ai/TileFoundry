"""A loop over an iterator that is neither tf.tile nor range."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="looping", target=cpu.Target())
class UnknownLoopIterator:
    @func
    def looping(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
        out = tf.add(seed, seed)
        for row in steps(1, 2):
            out = tf.add(x[row, :], seed)
        return out
