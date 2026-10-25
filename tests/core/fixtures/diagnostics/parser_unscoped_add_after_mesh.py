"""An op call after a mesh scope, outside every scope it opens."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="add_after_mesh", target=CpuTarget(), topologies=(Topology("cta", 1),))
class AddAfterMesh:
    @func
    def add_after_mesh(x: Tensor[(4, 4), "f32"]):
        with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
            _value = x
        result = tf.add(x, x)
        return result
