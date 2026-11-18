"""FSDP on two GPUs: the weight is split, gathered, and its gradient reduce-scattered."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6
target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("shard",))


def placed(shape, attr):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


A, Sharded, Full = placed((M, K), Split(0)), placed((N, K), Split(0)), placed((N, K), Broadcast())


@module(entry="forward", target=target, topologies=(gpu,))
class Layer:
    @func
    def forward(x: A, w: Sharded):
        full = tf.reshard(w, layout=Full.layout, storage="gmem")
        h = tf.matmul(x, full, b_layout="NK")
        return h * h
