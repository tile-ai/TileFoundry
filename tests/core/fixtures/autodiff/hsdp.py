"""HSDP on a (replicate, shard) mesh: one Reshard reduces the weight gradient on both axes."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 4),))
class Hsdp:
    @func(mesh=Mesh(("gpu",), (2, 2), names=("replicate", "shard")))
    def forward(x: Tensor[(M @ (mesh.replicate, mesh.shard), K), "f32", "gmem"],
                w: Tensor[(N @ mesh.shard, K), "f32", "gmem"]):
        full = tf.reshard(w, layout=(N, K) @ {B()}, storage="gmem")
        h = tf.matmul(x, full, b_layout="NK")
        return h * h
