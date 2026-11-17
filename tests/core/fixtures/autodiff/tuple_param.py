"""Weights passed as one tuple parameter, read by index."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class TupleParam:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: tuple[Tensor[(3, 3), "f32"], tuple[Tensor[(3,), "f32"]]]):
        h = tf.matmul(x, w[0]) + w[1][0]
        return h * h
