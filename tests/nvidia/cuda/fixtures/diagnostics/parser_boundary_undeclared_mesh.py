from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="root",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class UndeclaredMesh:
    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def root(x: Tensor[(8 @ nope, 8), "f32"]):
        return tf.add(x, x)
