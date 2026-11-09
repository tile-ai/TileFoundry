from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="root",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class StorageMismatch:
    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def stage(x: Tensor[(8 @ mesh.b, 8), "f32", "rmem"]):
        return tf.add(x, x)

    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def root(x: Tensor[(8, 8), "f32"]):
        return stage(
            tf.reshard(x, (8 @ mesh.b, 8), "smem")
        )
