"""A cta root that reaches a child declaring a warp topology hierarchy."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_H200 = "nvidia.h200_sxm"
_CTA = (Topology("cta", 132),)


@module(entry="run", topologies=(Topology("warp", 4),))
class _Warped:
    @func
    def run(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return tf.add(x, x)


@module(entry="fused", target=CudaTarget(_H200), topologies=_CTA)
class _Mismatch:
    warped = _Warped

    @func
    def fused(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return warped(x)  # noqa: F821
