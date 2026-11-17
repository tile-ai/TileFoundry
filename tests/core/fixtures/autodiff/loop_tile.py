"""A tf.tile loop: the induction value is a coordinate that starts each window of x."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class LoopTile:
    @func
    def forward(x: Tensor[(8, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = tf.zeros(Tensor[(2, 3), "f32"])
        for t in tf.tile(8, 2):
            h = tf.matmul(h * h + x[t, :], w)
        return h
