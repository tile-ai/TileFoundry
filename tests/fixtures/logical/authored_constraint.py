"""An elementwise operation whose placement is deferred to scheduling."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(
    entry="constrained",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class AuthoredConstraint:
    @func
    def constrained(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
        cta_mesh = Mesh((Topology("cta", 8),), Layout((8,), (1,)))
        y: where(layout=(8 @ cta, 16), mesh=cta_mesh, storage="gmem") = tf.add(x, x)
        return y
