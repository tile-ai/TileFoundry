"""An elementwise operation whose placement is deferred to scheduling."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.target import CudaTarget

_CTA_MESH = Mesh((Topology("cta", 8),), Layout((8,), (1,)))


@module(
    entry="constrained",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 8),),
)
class AuthoredConstraint:
    @func
    def constrained(x: Tensor[(8, 16), "bf16"]) -> Tensor[(8, 16), "bf16"]:
        y: where(layout=(8 @ cta, 16), mesh=_CTA_MESH, storage="gmem") = tf.add(x, x)
        return y
