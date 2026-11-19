"""Two pipeline stages beside a value every GPU holds."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu = Topology("gpu", 2)
gpus = Mesh((gpu,), Layout((2,)), ("pp",))


def on(units, shape):
    return TensorType(shape, DType.f32, ShardLayout.from_logical(
        shape, units, (Broadcast(),), storage="gmem", owners=target.storage_owners), "gmem")


H1 = on(gpus[1], (4, 8))


@module(entry="forward", target=target, topologies=(gpu, Topology("cta", 1)))
class SharedWeight:
    @func(mesh=Mesh(("gpu",), (2,), names=("pp",)))
    def forward(
        x: on(gpus[0], (4, 8)),
        w0: on(gpus[0], (8, 8)),
        shared: Tensor[(8, 8), "f32", (8, 8) @ {B()}, "gmem"],
        w1: on(gpus[1], (8, 16)),
    ):
        with Mesh(mesh[0], layout=(1,), names=("s",)) as _s0:
            h = tf.matmul(x, w0)
        moved = tf.reshard(h, H1.layout, "gmem")
        with Mesh(mesh[1], layout=(1,), names=("s",)) as _s1:
            y = tf.matmul(moved, w1)
        scaled = shared * 2.0
        return y, scaled
