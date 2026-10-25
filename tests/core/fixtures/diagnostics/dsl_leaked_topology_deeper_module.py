"""A Module declared one level deeper than a raising class body, using that body's topology."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget

try:

    @module(topologies=(Topology("leaked", 1),))
    class Failed:
        raise RuntimeError("class body failed")
except RuntimeError:
    pass


def define_one_level_deeper():
    @module(entry="probe", target=CpuTarget())
    class Later:
        @func
        def probe(x: Tensor[(1,), "f32"]) -> Tensor[(1,), "f32"]:
            with Mesh(("leaked",), (1,), ("lane",)) as _mesh:
                return tf.relu(x)

    return Later


Later = define_one_level_deeper()
