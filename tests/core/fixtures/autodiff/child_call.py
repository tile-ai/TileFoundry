"""A Module that calls the entry of a child slot and of an indexed child slot."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward")
class Layer:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = tf.matmul(x, w)
        return h * tf.exp(-h)


@module(entry="forward", target=cpu.Target())
class ChildCall:
    first = Layer
    rest = ModuleList(Layer, 2)

    @func
    def forward(x: Tensor[(4, 3), "f32"], w1: Tensor[(3, 3), "f32"], w2: Tensor[(3, 3), "f32"]):
        y = first(x, w1)
        z = rest[0](y, w2)
        return rest[1](z, w1)
