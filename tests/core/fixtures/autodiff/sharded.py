"""FSDP on two GPUs: the weight is split, gathered, and its gradient reduce-scattered."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class Layer:
    mesh = Mesh(("gpu",), (2,), names=("shard",))

    @func
    def forward(x: Tensor[(M @ mesh.shard, K), "f32", "gmem"], w: Tensor[(N @ mesh.shard, K), "f32", "gmem"]):
        with mesh:
            full = tf.reshard(w, layout=(N, K), storage="gmem")
            h = tf.matmul(x, full, b_layout="NK")
            return h * h
