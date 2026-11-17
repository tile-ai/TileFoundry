"""A loop whose trip count is a dimension of the input, not a number."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

N = DimVar("N", 1, 64)


@module(entry="forward", target=cpu.Target())
class LoopDynamic:
    @func
    def forward(x: Tensor[(N, 3), "f32"], h0: Tensor[(1, 3), "f32"]):
        h = h0
        for t in range(N):
            h = h * tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
        return h
