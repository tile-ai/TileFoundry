"""A TIR signature names a mesh the class body declares, as a HIR signature does."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(topologies=(Topology("cta", 1), Topology("thread", 8)))
class ClassMeshPrimFunc:
    lanes = Mesh(("thread",), (8,), names=("lane",))

    @prim_func(target=cuda.Target("nvidia.h200_sxm"))
    def clear(x: Tensor[(8, 16), "f32", (8 @ lanes.lane, 16), "rmem"]):
        with lanes as lane:
            T.fill(x, 0.0)
