"""A loop inside a loop: the inner loop's stacks are stacked again by the outer loop."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class LoopNested:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3,), "f32"], h0: Tensor[(1, 3), "f32"]):
        h = h0
        for t in range(4):
            i = tf.reshape(t, new_shape=(1,))
            for _ in range(2):
                h = h * w + tf.index_select(x, i, dim=0)
        return h * h
