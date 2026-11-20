"""FSDP over three layers in a loop: each layer's slice of the split stacked weights is gathered."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

L, T, C, F = 3, 8, 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class FsdpLayers:
    @func(mesh=Mesh(("gpu",), (2,), names=("shard",)))
    def layer(h: Tensor[(T @ mesh.shard, C), "f32", "gmem"], w1: Tensor[(F, C), "f32", (F, C), "gmem"],
              w2: Tensor[(C, F), "f32", (C, F), "gmem"]):
        return h + tf.matmul(tf.silu(tf.matmul(h, w1, b_layout="NK")), w2, b_layout="NK")

    @func(mesh=Mesh(("gpu",), (2,), names=("shard",)))
    def forward(x: Tensor[(T @ mesh.shard, C), "f32", "gmem"], up: Tensor[(L, F @ mesh.shard, C), "f32", "gmem"],
                down: Tensor[(L, C, F @ mesh.shard), "f32", "gmem"]):
        h = x
        for t in range(L):
            index = tf.reshape(t, new_shape=(1,))
            w1 = tf.reshard(tf.reshape(tf.index_select(up, index, dim=0), new_shape=(F, C)),
                            layout=(F, C), storage="gmem")
            w2 = tf.reshard(tf.reshape(tf.index_select(down, index, dim=0), new_shape=(C, F)),
                            layout=(C, F), storage="gmem")
            h = layer(h, w1, w2)
        return h * h
