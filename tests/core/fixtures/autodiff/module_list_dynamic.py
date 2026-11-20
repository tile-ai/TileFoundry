"""A loop over the elements of a ModuleList whose trip count is a dimension."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

N = DimVar("N", 1, 3)


@module(entry="forward")
class Layer:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = tf.matmul(x, w)
        return h * tf.exp(-h)


@module(entry="forward", target=cpu.Target())
class Stack:
    layers = ModuleList(Layer, 3)

    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"], n: Tensor[(N,), "f32"]):
        h = x
        for t in range(N):
            h = layers[t](h, w)
        return h
