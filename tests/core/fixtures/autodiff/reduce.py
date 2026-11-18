"""SUM, MEAN and MAX reductions, kept and dropped axes, and a maximum that ties."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Reduce:
    @func
    def forward(x: Tensor[(4, 3, 2), "f32"]):
        s = tf.reduce(x * x, axes=(1,), kind=ReduceKind.SUM)
        m = tf.reduce(x, axes=(0, 2), kind=ReduceKind.MEAN, keepdim=False)
        tie = tf.reduce(tf.maximum(x, 0.25), axes=(-1,), kind=ReduceKind.MAX)
        top = tf.reduce(x, axes=(0,), kind=ReduceKind.MAX, keepdim=False)
        return s * tie, m, top
