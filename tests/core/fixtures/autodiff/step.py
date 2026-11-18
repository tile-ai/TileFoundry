"""A training step written by hand around a generated forward and backward (D27)."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="forward", target=cpu.Target())
class Layer:
    @func
    def forward(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = tf.matmul(x, w)
        return h * tf.exp(-h)


LayerForward, LayerBackward = differentiate(Layer, wrt=("w",))


@module(entry="step", target=cpu.Target())
class Training:
    layer = LayerForward
    layer_backward = LayerBackward

    @func
    def step(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"], target: Tensor[(4, 3), "f32"]):
        y, saved = layer(x, w)
        error = y - target
        loss = tf.reduce(error * error, axes=(0, 1), kind=ReduceKind.SUM, keepdim=False)
        gw = layer_backward(2.0 * error, saved)
        return loss, w - 0.1 * gw
