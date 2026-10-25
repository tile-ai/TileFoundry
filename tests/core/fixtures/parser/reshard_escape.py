"""A thread-placed value that only a reshard carries into a CTA scope."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(
    entry="run",
    target=CpuTarget(),
    topologies=(Topology("cta", 2), Topology("thread", 4)),
)
class ReshardEscape:
    @func
    def run(x: Tensor[(8,), "f32"]):
        with Mesh(("thread",), (4,), names=("lane",)) as lanes:
            value = tf.reshard(x, (8 @ lanes.lane,), "rmem")
        with Mesh(("cta",), (2,), names=("block",)) as blocks:
            return tf.reshard(value, (8 @ blocks.block,), "smem")
