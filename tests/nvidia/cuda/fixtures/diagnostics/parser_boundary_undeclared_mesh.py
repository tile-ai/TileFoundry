from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="root",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class UndeclaredMesh:
    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def root(x: Tensor[(8 @ nope, 8), "f32"]):  # noqa: F821
        return tf.add(x, x)
