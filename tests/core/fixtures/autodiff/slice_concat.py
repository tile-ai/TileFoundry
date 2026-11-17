"""A window of the input, and a concatenation of two pieces."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class SliceConcat:
    @func
    def forward(x: Tensor[(4, 6), "f32"], y: Tensor[(4, 2), "f32"]):
        window = x[1:3, 2:5]
        joined = tf.concat([x, y * y], axis=1)
        return tf.reduce(window, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False) * joined
