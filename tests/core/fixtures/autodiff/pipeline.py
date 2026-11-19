"""Two pipeline stages, each on one slice of the mesh, joined by a Reshard over the whole mesh."""

from tilefoundry.dsl import *
from tilefoundry.target import cpu

target = cpu.Target()
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("pp",))


def on(stage, shape):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, stage, (Broadcast(),), storage="gmem", owners=target.storage_owners), "gmem")


H1 = on(mesh[1], (4, 8))


@module(entry="forward", target=target, topologies=(gpu,))
class Pipeline:
    @func
    def forward(x: on(mesh[0], (4, 8)), w0: on(mesh[0], (8, 8)), w1: on(mesh[1], (8, 8))):
        with Mesh(("gpu",), layout=(2,), names=("pp",)) as stages:
            with Mesh(stages[0], layout=(1,), names=("s",)) as _s0:
                h = tf.matmul(x, w0)
            moved = tf.reshard(h, H1.layout, "gmem")
            with Mesh(stages[1], layout=(1,), names=("s",)) as _s1:
                return tf.matmul(moved, w1)
