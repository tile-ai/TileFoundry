"""Tanh has no autodiff rule."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class NoRule:
    @func
    def forward(x: Tensor[(4, 3), "f32"]):
        r = tf.tanh(x)
        return r * x
