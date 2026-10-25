"""A return expression after a mesh scope, outside every scope it opens."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="run", target=CpuTarget(), topologies=(Topology("cta", 1),))
class UnscopedReturn:
    @func
    def run(x: Tensor[(4,), "f32"]):
        with Mesh(("cta",), (1,), names=("unit",)) as _mesh:
            value = x
        return tf.add(value, value)
