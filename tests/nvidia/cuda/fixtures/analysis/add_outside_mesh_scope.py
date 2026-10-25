"""An add written outside any mesh scope."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_TARGET = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 2), Topology("thread", 4))


@module(entry="f", target=_TARGET, topologies=_TOPOLOGIES)
class NoScope:
    @func
    def f(x: Tensor[(8, 16), "f32"]):
        return x + x
