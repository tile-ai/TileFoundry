"""A range loop with one carry, reading one row of x per step and a loop-invariant weight."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class LoopRange:
    @func
    def forward(x: Tensor[(5, 3), "f32"], w: Tensor[(3,), "f32"], h0: Tensor[(1, 3), "f32"]):
        h = h0
        for t in range(5):
            row = tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            h = tf.exp(-h * w) * h + row
        return h
