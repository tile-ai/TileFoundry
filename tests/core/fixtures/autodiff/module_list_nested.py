"""The loop over the elements of a ModuleList runs inside a loop over two microbatches."""

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
    def forward(x: Tensor[(4, 3), "f32"], ws: Tensor[(3, 3, 3), "f32"]):
        h = x
        for m in range(2):
            for t in range(3):
                w = tf.reshape(tf.index_select(ws, tf.reshape(t, new_shape=(1,)), dim=0), new_shape=(3, 3))
                h = layers[t](h, w)
        return h * h
