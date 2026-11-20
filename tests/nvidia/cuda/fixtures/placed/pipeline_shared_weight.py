"""Two pipeline stages beside a value every GPU holds."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu = Topology("gpu", 2)


@module(entry="forward", target=target, topologies=(gpu, Topology("cta", 1)))
class SharedWeight:
    @func(mesh=Mesh(("gpu",), (2,), names=("pp",)))
    def forward(
        x: Tensor[(4, 8), "f32", (4, 8) @ mesh[0], "gmem"],
        w0: Tensor[(8, 8), "f32", (8, 8) @ mesh[0], "gmem"],
        shared: Tensor[(8, 8), "f32", (8, 8), "gmem"],
        w1: Tensor[(8, 16), "f32", (8, 16) @ mesh[1], "gmem"],
    ):
        with Mesh(mesh[0], layout=(1,), names=("s",)) as _s0:
            h = tf.matmul(x, w0)
        moved = tf.reshard(h, (4, 8) @ mesh[1], "gmem")
        with Mesh(mesh[1], layout=(1,), names=("s",)) as _s1:
            y = tf.matmul(moved, w1)
        scaled = shared * 2.0
        return y, scaled
