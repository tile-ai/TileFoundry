"""IndexSelect of a table split on the axis it does not index; the backward writes rows into it."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

target = cpu.Target()
gpu = Topology("gpu", 2)


@module(entry="forward", target=target, topologies=(gpu,))
class SplitIndexSelect:
    gpu = Mesh(("gpu",), (2,), names=("shard",))

    @func
    def forward(
        table: Tensor[(5, 6 @ gpu.shard), "f32"], index: Tensor[(3,), "i64"]
    ):
        rows = tf.index_select(table, index, dim=0)
        return rows * rows
