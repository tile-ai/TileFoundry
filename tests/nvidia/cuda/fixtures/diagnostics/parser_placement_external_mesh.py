from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_EXTERNAL_PLACEMENT_MESH = Mesh(("cta",), (2,), names=("tile",))


@module(
    entry="run",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 2),),
)
class ExternalMeshPlacement:
    @func
    def run(x: Tensor[(2 @ _EXTERNAL_PLACEMENT_MESH.tile,), "f32"]):  # noqa: F821
        return x
