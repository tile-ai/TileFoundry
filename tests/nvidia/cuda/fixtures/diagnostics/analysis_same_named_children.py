"""Two different child Modules that are both named Expert, under one root."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(entry="run")
class Expert:
    @func
    def run(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return x * 2.0


LeftExpert = Expert


@module(entry="run")
class Expert:
    @func
    def run(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return x + 1.0


@module(entry="root", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 132),))
class Root:
    left = LeftExpert
    right = Expert

    @func
    def root(x: Tensor[(4, 8), "f32"]) -> Tensor[(4, 8), "f32"]:
        return right(left(x))
