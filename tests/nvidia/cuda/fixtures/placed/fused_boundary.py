"""Three call edges share one recursively matched boundary contract."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="run",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8), Topology("thread", 32)),
)
class Inner:
    mesh = Mesh(("cta",), (8,), names=("b",))

    @func
    def scale(
        x: Tensor[(8, 8), "bf16", None, "rmem"],
        w: Tensor[(8, 8), "bf16", None, "umat"],
    ):
        with mesh:
            return tf.mul(x, w)

    @func
    def run(x: Tensor[(8, 8 @ mesh.b), "bf16", "rmem"]):
        with mesh:
            return scale(x, x)


@module(
    entry="root",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8), Topology("thread", 32)),
)
class FusedBoundary:
    mesh = Mesh(("cta",), (8,), names=("b",))

    inner = Inner

    @func
    def stage(x: Tensor[(8 @ mesh.b, 8), "f32", "smem"]):
        with mesh:
            with Mesh(("thread",), (32,), names=("t",)) as _t:
                return tf.add(x, x)

    @func
    def root(x: Tensor[(8, 8), "f32"]):
        with mesh:
            a = tf.reshard(x, (8 @ mesh.b, 8), "smem")
            y = stage(a)
            b = tf.reshard(
                tf.cast(y, dtype="bf16"), (8, 8 @ mesh.b), "rmem"
            )
            z = inner(b)
            return tf.reshard(tf.cast(z, dtype="f32"), (8, 8), "gmem")
