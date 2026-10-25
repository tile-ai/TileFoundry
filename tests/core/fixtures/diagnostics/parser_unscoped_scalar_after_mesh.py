"""Scalar arithmetic after a mesh scope, outside every scope it opens."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="scalar_after_mesh", target=CpuTarget(), topologies=(Topology("cta", 1),))
class ScalarAfterMesh:
    @func
    def scalar_after_mesh(x: Tensor[(4, 4), "f32"]):
        with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
            _value = x
        result = 1 + 2
        return result
