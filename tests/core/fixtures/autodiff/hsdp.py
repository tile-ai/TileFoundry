"""HSDP on a (replicate, shard) mesh: one Reshard reduces the weight gradient on both axes."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

M, K, N = 8, 4, 6
target = cpu.Target()
gpu = Topology("gpu", 4)
mesh = Mesh((gpu,), Layout((2, 2)), ("replicate", "shard"))


def placed(shape, attrs):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, attrs, storage="gmem", owners=target.storage_owners), "gmem")


A = placed((M, K), (Split(0), Split(0)))
Sharded, Full = placed((N, K), (Broadcast(), Split(0))), placed((N, K), (Broadcast(), Broadcast()))


@module(entry="forward", target=target, topologies=(gpu,))
class Hsdp:
    @func
    def forward(x: A, w: Sharded):
        full = tf.reshard(w, layout=Full.layout, storage="gmem")
        h = tf.matmul(x, full, b_layout="NK")
        return h * h
