"""An op call before a mesh scope, outside every scope it opens."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="add_before_mesh", target=CpuTarget(), topologies=(Topology("cta", 1),))
class AddBeforeMesh:
    @func
    def add_before_mesh(x: Tensor[(4, 4), "f32"]):
        result = tf.add(x, x)
        with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
            _value = x
        return result
