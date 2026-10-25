from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

size = DimVar("prototype_mismatch_size", 1, 8)


@module(
    entry="root",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class IncompatibleVariant:
    @func
    def root(
        x: Tensor[(size,), "f32"],
    ) -> tuple[Tensor[(size,), "f32"], Tensor[(size,), "f32"]]:
        pass

    @root.specialize(RangePattern("prototype_mismatch_size", 1, 8))
    def scalar(x: Tensor[(size,), "f32"]):
        return x
