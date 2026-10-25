from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="root",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class LayoutUndefinedStorageMismatch:
    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def stage(x: Tensor[(8, 8), "bf16", None, "smem"]):
        return tf.add(x, x)

    @func(mesh=Mesh(("cta",), (8,), names=("b",)))
    def root(x: Tensor[(8, 8), "f32"]):
        actual = tf.reshard(tf.cast(x, dtype="bf16"), (8, 8), "rmem")
        return stage(actual)  # noqa: F821
