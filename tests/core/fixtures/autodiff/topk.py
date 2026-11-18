"""The largest values of each row; the indices are not differentiated."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Topk:
    @func
    def forward(x: Tensor[(4, 6), "f32"]):
        values, indices = tf.topk(x, k=2, axis=-1)
        return values * values
