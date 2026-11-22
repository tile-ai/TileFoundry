"""A layer gathers its split weight and passes the whole weight to a child Module.

The child's backward reads the weight it was given. The layer made that weight,
so the layer decides whether the forward keeps it or the backward gathers again.
A child that reshapes the weight first saves the reshape, which a selector names
on its own.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

T, C = 8, 4


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class Kernel:
    mesh = Mesh(("gpu",), (2,), names=("shard",))

    @func
    def forward(x: Tensor[(T @ mesh.shard, C), "f32", "gmem"], w: Tensor[(C, C), "f32", "gmem"]):
        with mesh:
            return tf.exp(tf.matmul(x, w))


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class GatheredLayer:
    mesh = Mesh(("gpu",), (2,), names=("shard",))
    kernel = Kernel

    @func
    def forward(x: Tensor[(T @ mesh.shard, C), "f32", "gmem"], w: Tensor[(C @ mesh.shard, C), "f32", "gmem"]):
        with mesh:
            w_full = tf.reshard(w, layout=(C, C), storage="gmem")
            return kernel(x, w_full)


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class ReshapingKernel:
    mesh = Mesh(("gpu",), (2,), names=("shard",))

    @func
    def forward(x: Tensor[(T @ mesh.shard, C), "f32", "gmem"], w: Tensor[(C * C,), "f32", "gmem"]):
        with mesh:
            w_view = tf.reshape(w, new_shape=(C, C))
            return tf.exp(tf.matmul(x, w_view))


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class ReshapedLayer:
    mesh = Mesh(("gpu",), (2,), names=("shard",))
    kernel = ReshapingKernel

    @func
    def forward(x: Tensor[(T @ mesh.shard, C), "f32", "gmem"], w: Tensor[((C * C) @ mesh.shard,), "f32", "gmem"]):
        with mesh:
            w_full = tf.reshard(w, layout=(C * C,), storage="gmem")
            return kernel(x, w_full)
