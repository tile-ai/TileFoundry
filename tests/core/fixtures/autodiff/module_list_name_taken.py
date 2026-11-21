"""A function of the root already has the name of the first segment of its ModuleList."""

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
    def layers_0(x: Tensor[(4, 3), "f32"]):
        return x * x

    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = layers_0(x)
        for t in range(3):
            h = layers[t](h, w)
        return h
