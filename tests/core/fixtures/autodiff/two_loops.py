"""Two loops one after the other: the second carries what the first returns."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class TwoLoops:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = x
        for t in range(2):
            h = tf.matmul(h, w)
        for s in range(2):
            h = tf.matmul(h, w)
        return h
