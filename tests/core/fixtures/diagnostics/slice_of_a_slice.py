"""A slice of a slice: a type slices only the mesh its name binds."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu


@module(entry="run", target=cpu.Target(), topologies=(Topology("gpu", 4),))
class SliceOfASlice:
    @func
    def run(x: Tensor[(4, 8), "f32"]):
        with Mesh(("gpu",), layout=(4,), names=("g",)) as units:
            with units[0:2] as first:
                return tf.reshard(x, (4, 8) @ first[0:1], "gmem")
