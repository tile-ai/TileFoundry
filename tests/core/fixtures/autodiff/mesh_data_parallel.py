"""Data parallel with one Mesh region around the whole body: the backward opens the same mesh."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class MeshDataParallel:
    @func(mesh=Mesh(("gpu",), (2,), names=("data",)))
    def forward(x: Tensor[(M @ mesh.data, K), "f32", "gmem"], w: Tensor[(K, N), "f32", (K, N) @ {B()}, "gmem"]):
        h = tf.matmul(x, w)
        return h * h
