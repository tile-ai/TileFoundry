"""A detached tuple projection used to exercise source diagnostics."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


@module(
    entry="project",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1),),
)
class TupleProjectionDiagnostic:
    @func
    def project(x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 1), "f32"]:
        with Mesh(("cta",), layout=(1,), names=("unit",)) as _mesh:
            pair = tf.topk(x, k=1, axis=-1)
        values, indices = pair
        return values
