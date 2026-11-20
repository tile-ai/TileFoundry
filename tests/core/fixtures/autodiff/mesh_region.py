"""Two Mesh regions: the values the backward reads leave each region as extra results."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class MeshBody:
    mesh = Mesh(("gpu",), (2,), names=("shard",))

    @func
    def forward(x: Tensor[(M @ mesh.shard, K), "f32", "gmem"], w: Tensor[(N @ mesh.shard, K), "f32", "gmem"]):
        with mesh:
            with Mesh(("gpu",), layout=(2,), names=("shard",)) as _gather:
                full = tf.reshard(w, layout=(N, K), storage="gmem")
                h = tf.matmul(x, full, b_layout="NK")
            with Mesh(("gpu",), layout=(2,), names=("shard",)) as _square:
                return h * tf.exp(h)
