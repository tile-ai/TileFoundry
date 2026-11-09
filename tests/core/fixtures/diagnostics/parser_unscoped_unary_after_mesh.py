"""A unary operator after a mesh scope, outside every scope it opens."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="unary_after_mesh", target=cpu.Target(), topologies=(Topology("cta", 1),))
class UnaryAfterMesh:
    @func
    def unary_after_mesh(x: Tensor[(4, 4), "f32"]):
        with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
            _value = x
        result = -x
        return result
