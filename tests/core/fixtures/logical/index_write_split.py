"""IndexAdd and IndexCopy into a table split on the axis the index does not address."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import cpu

target = cpu.Target()
gpu = Topology("gpu", 2)


@module(entry="forward", target=target, topologies=(gpu,))
class IndexAddColumns:
    gpu = Mesh(("gpu",), (2,), names=("shard",))

    @func
    def forward(
        table: Tensor[(5, 6 @ gpu.shard), "f32"],
        index: Tensor[(3,), "i64"],
        rows: Tensor[(3, 6 @ gpu.shard), "f32"],
    ):
        return tf.index_add(table, index, rows, dim=0)


@module(entry="forward", target=target, topologies=(gpu,))
class IndexCopyColumns:
    gpu = Mesh(("gpu",), (2,), names=("shard",))

    @func
    def forward(
        table: Tensor[(5, 6 @ gpu.shard), "f32"],
        index: Tensor[(3,), "i64"],
        rows: Tensor[(3, 6 @ gpu.shard), "f32"],
    ):
        return tf.index_copy(table, index, rows, dim=0)
