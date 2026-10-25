"""Three indexed ScaledChild layers, the loop reaching only the first two."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.shapes.scaled_modules import ScaledChild
from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_H200 = "nvidia.h200_sxm"
_CTA = (Topology("cta", 132),)
count = 3
trips = 2


@module(entry="run", target=CudaTarget(_H200), topologies=_CTA)
class Stack:
    layers = ModuleList(ScaledChild, count)

    @func
    def run(x: Tensor[(4,), "f32"]) -> Tensor[(4,), "f32"]:
        for i in range(trips):
            x = layers[i](x)  # noqa: F821 -- class-body binding
        return x
