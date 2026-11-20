"""A layout naming two slices of one mesh: one type holds one unit set."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="run", target=cpu.Target(), topologies=(Topology("gpu", 4),))
class TwoUnitSets:
    @func(mesh=Mesh(("gpu",), (2, 2), names=("pp", "cp")))
    def run(x: Tensor[(8, 16), "f32", (8 @ mesh[0].cp, 16) @ {mesh[1].pp @ V()}]):
        return x
