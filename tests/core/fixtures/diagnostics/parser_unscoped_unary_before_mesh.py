"""A unary operator before a mesh scope, outside every scope it opens."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="unary_before_mesh", target=CpuTarget(), topologies=(Topology("cta", 1),))
class UnaryBeforeMesh:
    @func
    def unary_before_mesh(x: Tensor[(4, 4), "f32"]):
        result = -x
        with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
            _value = x
        return result
