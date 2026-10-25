"""A call to a sibling function before a mesh scope, outside every scope it opens."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="call_before_mesh", target=CpuTarget(), topologies=(Topology("cta", 1),))
class CallBeforeMesh:
    @func
    def mesh_scope_helper(x: Tensor[(4, 4), "f32"]):
        return x

    @func
    def call_before_mesh(x: Tensor[(4, 4), "f32"]):
        result = mesh_scope_helper(x)
        with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
            _value = x
        return result
