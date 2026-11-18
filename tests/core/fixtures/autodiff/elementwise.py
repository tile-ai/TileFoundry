"""Pointwise ops with broadcast inputs: each Binary and Unary kind that has a rule."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Elementwise:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3,), "f32"], b: Tensor[(4, 1), "f32"]):
        h = x * w + b
        e = tf.exp(-h) / (h * h + 1.0)
        m = tf.maximum(e, h - x)
        return tf.rsqrt(tf.square(m) + 1.0) - m
