from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="root",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class AxisMismatch:
    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def stage(x: Tensor[(8, 8 @ mesh.b), "f32", "smem"]):  # noqa: F821
        return tf.add(x, x)

    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def root(x: Tensor[(8, 8), "f32"]):
        return stage(  # noqa: F821
            tf.reshard(x, (8 @ mesh.b, 8), "smem")  # noqa: F821
        )
