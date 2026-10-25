"""Integer arithmetic in a body names a sharded width before the layout is built."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget


@module(
    entry="f",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 128),),
)
class Model:
    @func
    def f(
        x: Tensor[(1, 16, 8192), "f32"],
    ):
        with Mesh(("cta",), layout=(128,), names=("unit",)) as mesh:
            width = 4096 + 4096
            return tf.reshard(
                x,
                (1, 16, width @ mesh.unit),
                "smem",
            )
