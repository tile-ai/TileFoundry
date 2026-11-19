"""Two pipeline stages beside a value every GPU holds."""

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu = Topology("gpu", 2)
mesh = Mesh((gpu,), Layout((2,)), ("pp",))


def on(units, shape):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, units, (Broadcast(),), storage="gmem", owners=target.storage_owners), "gmem")


H1 = on(mesh[1], (4, 8))


@module(entry="forward", target=target, topologies=(gpu, Topology("cta", 1)))
class SharedWeight:
    @func
    def forward(
        x: on(mesh[0], (4, 8)),
        w0: on(mesh[0], (8, 8)),
        shared: on(mesh, (8, 8)),
        w1: on(mesh[1], (8, 16)),
    ):
        with Mesh(("gpu",), layout=(2,), names=("pp",)) as stages:
            with Mesh(stages[0], layout=(1,), names=("s",)) as _s0:
                h = tf.matmul(x, w0)
            moved = tf.reshard(h, H1.layout, "gmem")
            with Mesh(stages[1], layout=(1,), names=("s",)) as _s1:
                y = tf.matmul(moved, w1)
            scaled = shared * 2.0
            return y, scaled
