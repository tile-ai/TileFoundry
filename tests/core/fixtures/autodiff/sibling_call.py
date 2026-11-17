"""A function that calls another function of its Module twice; the callee returns a tuple."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class SiblingCall:
    @func
    def layer(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = tf.matmul(x, w)
        return h * h, h

    @func
    def forward(x: Tensor[(4, 3), "f32"], w1: Tensor[(3, 3), "f32"], w2: Tensor[(3, 3), "f32"]):
        a, ha = layer(x, w1)
        b, hb = layer(a, w2)
        return b + ha
