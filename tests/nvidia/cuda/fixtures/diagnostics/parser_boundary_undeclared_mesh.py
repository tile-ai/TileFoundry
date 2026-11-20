from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="root",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class UndeclaredMesh:
    mesh = Mesh(("cta",), (8,), names=("b",))

    @func
    def root(x: Tensor[(8 @ nope, 8), "f32"]):
        with mesh:
            return tf.add(x, x)
