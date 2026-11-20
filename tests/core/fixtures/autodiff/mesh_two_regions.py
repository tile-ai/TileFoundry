"""Two Mesh regions that both read x and w: their gradients are added inside the mesh."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6


@module(entry="forward", target=cpu.Target(), topologies=(Topology("gpu", 2),))
class MeshTwoRegions:
    @func(mesh=Mesh(("gpu",), (2,), names=("data",)))
    def forward(x: Tensor[(M @ mesh.data, K), "f32", "gmem"], w: Tensor[(K, N), "f32", (K, N), "gmem"]):
        with Mesh(("gpu",), layout=(2,), names=("data",)) as _a:
            h = tf.matmul(x, w)
        with Mesh(("gpu",), layout=(2,), names=("data",)) as _b:
            return h * tf.exp(h) + tf.matmul(x, w)
