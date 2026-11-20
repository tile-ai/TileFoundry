"""Data parallel on two GPUs: the batch is split, so the weight gradient is all-reduced."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class DataParallel:
    mesh = Mesh(("gpu",), (2,), names=("data",))

    @func
    def forward(x: Tensor[(M @ mesh.data, K), "f32", "gmem"], w: Tensor[(K, N), "f32", (K, N), "gmem"]):
        with mesh:
            h = tf.matmul(x, w)
            return h * h
