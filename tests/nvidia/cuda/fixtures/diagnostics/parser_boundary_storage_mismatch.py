from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="root",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class StorageMismatch:
    mesh = Mesh(("cta",), (8,), names=("b",))

    @func
    def stage(x: Tensor[(8 @ mesh.b, 8), "f32", "rmem"]):
        with mesh:
            return tf.add(x, x)

    @func
    def root(x: Tensor[(8, 8), "f32"]):
        with mesh:
            return stage(
                tf.reshard(x, (8 @ mesh.b, 8), "smem")
            )
