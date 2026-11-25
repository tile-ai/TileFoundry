"""A training step that calls a generated forward and backward whose loops call a layer function."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

_H200 = cuda.Target("nvidia.h200_sxm")
_CTA = (Topology("cta", 132),)


@module(entry="forward", target=_H200, topologies=_CTA)
class Layers:
    @func
    def layer(x: Tensor[(4, 3), "f32"], w: Tensor[(3, 3), "f32"]):
        h = tf.matmul(x, w)
        return h * tf.exp(-h)

    @func
    def forward(x: Tensor[(4, 3), "f32"], ws: Tensor[(3, 3, 3), "f32"]):
        h = x
        for t in range(3):
            w = tf.reshape(tf.index_select(ws, tf.reshape(t, new_shape=(1,)), dim=0), new_shape=(3, 3))
            h = layer(h, w)
        return h


LayersForward, LayersBackward = differentiate(Layers, wrt=("ws",))


@module(entry="step", target=_H200, topologies=_CTA)
class StepLayers:
    f = LayersForward
    b = LayersBackward

    @func
    def step(x: Tensor[(4, 3), "f32"], ws: Tensor[(3, 3, 3), "f32"]):
        y, saved = f(x, ws)
        return ws - 0.1 * b(2.0 * y, saved)
