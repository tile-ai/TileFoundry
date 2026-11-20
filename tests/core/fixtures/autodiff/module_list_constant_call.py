"""A loop calls elements 0 and 1 of a ModuleList; another function calls element 2."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


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
    def head(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        return layers[2](x, w)

    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = x
        for t in range(2):
            h = layers[t](h, w)
        return head(h, w)
