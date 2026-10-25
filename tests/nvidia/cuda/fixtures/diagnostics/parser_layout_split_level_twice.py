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
    def f(x: Tensor[(4, 64), "f32"]):
        with Mesh(("cta",), layout=(4, 32), names=("x", "y")) as first:
            with Mesh(("cta",), layout=(2, 64), names=("p", "q")) as second:
                return tf.reshard(
                    x,
                    (4 @ first.x, 64 @ second.q),
                    "gmem",
                )
