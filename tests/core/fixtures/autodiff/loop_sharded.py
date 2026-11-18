"""A recurrence over a batch split on two GPUs: each stack keeps the split, one axis further in."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("data",))


def placed(shape, attr):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, mesh, (attr,), storage="gmem", owners=target.storage_owners), "gmem")


@module(entry="forward", target=target, topologies=(gpu,))
class LoopSharded:
    @func
    def forward(h0: placed((4, 3), Split(0)), w: placed((3, 3), Broadcast())):
        h = h0
        for _ in range(3):
            h = tf.matmul(h * h, w)
        return h
