from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda


@module(
    entry="root",
    target=cuda.Target("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class LayoutUndefinedStorageMismatch:
    mesh = Mesh(("cta",), (8,), names=("b",))

    @func
    def stage(x: Tensor[(8, 8), "bf16", None, "smem"]):
        with mesh:
            return tf.add(x, x)

    @func
    def root(x: Tensor[(8, 8), "f32"]):
        with mesh:
            actual = tf.reshard(tf.cast(x, dtype="bf16"), (8, 8), "rmem")
            return stage(actual)
