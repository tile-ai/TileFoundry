"""Reshape and Transpose between two weights."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class ReshapeTranspose:
    @func
    def forward(x: Tensor[(2, 3, 4), "f32"], w: Tensor[(4, 6), "f32"]):
        t = tf.transpose(x, perm=(2, 0, 1))
        r = tf.reshape(t, new_shape=(4, 6))
        return tf.reshape(r * w, new_shape=(2, 12)) * 3.0
