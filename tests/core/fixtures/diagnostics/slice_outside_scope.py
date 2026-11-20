"""A body type naming units outside the scope it is written in."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="run", target=cpu.Target(), topologies=(Topology("gpu", 4),))
class SliceOutsideScope:
    @func
    def run(x: Tensor[(4, 8), "f32"]):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as units:
            with units[0:2]:
                return tf.reshard(x, (4, 8) @ units[2:4], "gmem")
