"""IndexSelect of a table split on the axis it does not index; the backward writes rows into it."""
from tilefoundry.dsl import *
from tilefoundry.target import cpu

target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("shard",))


def placed(shape, attr, dtype=DType.f32):
    return TensorType(shape, dtype, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


@module(entry="forward", target=target, topologies=(gpu,))
class SplitIndexSelect:
    @func
    def forward(table: placed((5, 6), Split(1)), index: placed((3,), Broadcast(), DType.i64)):
        rows = tf.index_select(table, index, dim=0)
        return rows * rows
