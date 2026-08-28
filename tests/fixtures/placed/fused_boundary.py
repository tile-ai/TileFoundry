"""Three call edges share one recursively matched boundary contract."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, Topology, tf
from tilefoundry.target import CudaTarget


@module(
    entry="run",
    topologies=(Topology("cta", 8), Topology("thread", 32)),
)
class _Inner:
    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def scale(
        x: Tensor[(8, 8), "bf16", None, "rmem"],
        w: Tensor[(8, 8), "bf16", None, "umat"],
    ):
        return tf.mul(x, w)

    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def run(x: Tensor[(8, 8 @ mesh.b), "bf16", "rmem"]):  # noqa: F821
        return scale(x, x)  # noqa: F821


@module(
    entry="root",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8), Topology("thread", 32)),
)
class FusedBoundary:
    inner = _Inner

    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def stage(x: Tensor[(8 @ mesh.b, 8), "f32", "smem"]):  # noqa: F821
        with Mesh(("thread",), (32,), names=("t",)) as _t:
            return tf.add(x, x)

    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def root(x: Tensor[(8, 8), "f32"]):
        a = tf.reshard(x, (8 @ mesh.b, 8), "smem")  # noqa: F821
        y = stage(a)  # noqa: F821
        b = tf.reshard(  # noqa: F821
            tf.cast(y, dtype="bf16"), (8, 8 @ mesh.b), "rmem"
        )
        z = inner(b)  # noqa: F821
        return tf.reshard(tf.cast(z, dtype="f32"), (8, 8), "gmem")
