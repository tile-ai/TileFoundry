"""Data parallel with one Mesh region around the whole body: the backward opens the same mesh."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6
target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("data",))


def placed(shape, attr):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


@module(entry="forward", target=target, topologies=(gpu,))
class MeshDataParallel:
    @func
    def forward(x: placed((M, K), Split(0)), w: placed((K, N), Broadcast())):
        with Mesh(("gpu",), layout=(2,), names=("data",)) as _data:
            h = tf.matmul(x, w)
            return h * h
