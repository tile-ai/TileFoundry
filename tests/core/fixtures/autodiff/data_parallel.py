"""Data parallel on two GPUs: the batch is split, so the weight gradient is all-reduced."""

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
class DataParallel:
    @func
    def forward(x: placed((M, K), Split(0)), w: placed((K, N), Broadcast())):
        h = tf.matmul(x, w)
        return h * h
