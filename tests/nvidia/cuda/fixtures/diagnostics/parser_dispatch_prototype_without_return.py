from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="root",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class NoReturn:
    @func
    def root(x: Tensor[(8, 16), "f32"]):
        pass
