"""Sigmoid, Silu and Softplus of one input."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Activations:
    @func
    def forward(x: Tensor[(4, 5), "f32"]):
        return tf.sigmoid(x) * tf.silu(x) + tf.softplus(x * 3.0)
