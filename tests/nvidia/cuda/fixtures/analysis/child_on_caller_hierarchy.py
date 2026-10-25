"""A root and a child that both declare the same symbolic cta hierarchy."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_H200 = "nvidia.h200_sxm"
extent = DimVar("child_topology_extent", 1, 132)
hierarchy = (Topology("cta", extent),)


@module(entry="run", topologies=hierarchy)
class _Declared:
    @func
    def run(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.add(x, x)


@module(entry="fused", target=CudaTarget(_H200), topologies=hierarchy)
class _Agreeing:
    declared = _Declared

    @func
    def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return declared(x)  # noqa: F821
