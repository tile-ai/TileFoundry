"""IndexAdd and IndexCopy into a table split on the axis the index does not address."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("shard",))


def placed(shape, attr, dtype=DType.f32):
    return TensorType(shape, dtype, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


Table, Rows, Index = placed((5, 6), Split(1)), placed((3, 6), Split(1)), placed((3,), Broadcast(), DType.i64)


@module(entry="forward", target=target, topologies=(gpu, Topology("cta", 1)))
class IndexAddColumns:
    @func
    def forward(table: Table, index: Index, rows: Rows):
        return tf.index_add(table, index, rows, dim=0)


@module(entry="forward", target=target, topologies=(gpu, Topology("cta", 1)))
class IndexCopyColumns:
    @func
    def forward(table: Table, index: Index, rows: Rows):
        return tf.index_copy(table, index, rows, dim=0)
