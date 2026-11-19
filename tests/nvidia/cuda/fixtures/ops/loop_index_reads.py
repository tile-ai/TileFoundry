"""One loop per way a loop index's bounds decide whether it is the same on every GPU.

DirectCoord and HiddenCoord pick at the GPU's own coordinate instead, read
directly or returned by a function, so the row differs per GPU.

Each loop picks a row of a Broadcast table at the index. On the GPU target the
accumulator is plain, so every loop types and the picked row shows how the
index reads. On the CPU target Broadcast and Varying read alike.
"""

from tilefoundry.dsl import *
from tilefoundry.target import cpu
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu, cta = Topology("gpu", 2), Topology("cta", 1)
mesh = Mesh((gpu,), Layout((2,)), ("g",))
N = DimVar("N", 1, 8)
TABLE = TensorType((8, 8), DType.f32, ShardLayout.from_logical(
    (8, 8), mesh, (Broadcast(),), storage="gmem", owners=target.storage_owners), "gmem")
COUNT = TensorType((), DType.i64, ShardLayout.from_logical(
    (), mesh, (Broadcast(),), storage="gmem", owners=target.storage_owners), "gmem")


@module(entry="forward", target=target, topologies=(gpu, cta))
class LiteralBounds:
    @func
    def forward(x: TABLE, acc0: Tensor[(1, 8), "f32", "gmem"]):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            acc = acc0
            for t in range(4):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class BroadcastCount:
    @func
    def forward(x: TABLE, acc0: Tensor[(1, 8), "f32", "gmem"], n: COUNT):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            acc = acc0
            for t in range(n):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class PlainCount:
    @func
    def forward(x: TABLE, acc0: Tensor[(1, 8), "f32", "gmem"], n: Tensor[(), "i64", "gmem"]):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            acc = acc0
            for t in range(n):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class DimVarExtent:
    @func
    def forward(x: TABLE, acc0: Tensor[(1, 8), "f32", "gmem"]):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            acc = acc0
            for t in range(N):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class OuterIndexStart:
    @func
    def forward(x: TABLE, acc0: Tensor[(1, 8), "f32", "gmem"]):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            acc = acc0
            for i in range(2):
                for t in range(i + 1, 4):
                    acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class DirectCoord:
    @func
    def forward(x: TABLE):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as m:
            return tf.index_select(x, tf.reshape(m.g, new_shape=(1,)), dim=0)


@module(entry="forward", target=target, topologies=(gpu, cta))
class HiddenCoord:
    @func
    def rank():
        with Mesh(("gpu",), layout=(2,), names=("g",)) as m:
            return tf.reshape(m.g, new_shape=(1,))

    @func
    def forward(x: TABLE):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            return tf.index_select(x, rank(), dim=0)


host = cpu.Target()
HOST_TABLE, HOST_ROW = (
    TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (Broadcast(),), storage="gmem", owners=host.storage_owners), "gmem")
    for shape in ((8, 8), (1, 8))
)


@module(entry="forward", target=host, topologies=(gpu,))
class HostDimVarExtent:
    @func
    def forward(x: HOST_TABLE, acc0: HOST_ROW):
        with Mesh(("gpu",), layout=(2,), names=("g",)) as _m:
            acc = acc0
            for t in range(N):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc
