"""A tuple element read before a mesh scope, outside every scope it opens."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="tuple_before_mesh", target=CpuTarget(), topologies=(Topology("cta", 1),))
class TupleBeforeMesh:
    @func
    def tuple_before_mesh(
        x: Tensor[(4, 4), "f32"],
        pair: tuple[Tensor[(4, 4), "f32"], Tensor[(4, 4), "f32"]],
    ):
        result = pair[0]
        with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
            _value = x
        return result
