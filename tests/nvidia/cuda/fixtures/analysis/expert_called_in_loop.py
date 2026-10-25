"""A root that calls the child Expert once per trip of a three-trip loop."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.logical.hir_composition import Expert
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_H200 = "nvidia.h200_sxm"
_CTA = (Topology("cta", 132),)


@module(entry="looped", target=CudaTarget(_H200), topologies=_CTA)
class _Looped:
    mm = Expert

    @func
    def looped(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        acc = x
        for _ in range(3):
            acc = mm(acc)  # noqa: F821
        return acc
