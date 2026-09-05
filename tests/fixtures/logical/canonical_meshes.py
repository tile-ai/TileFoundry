from __future__ import annotations

from tilefoundry.module import module
from tilefoundry import func
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401
from tilefoundry.ir.types.shard import (
    B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,
)

cta = Mesh((Topology("cta", 8),), Layout((8,), (1,)), names=('b',))
cta_2 = Mesh((Topology("cta", 8),), Layout((8,), (1,)), names=('b',))

@module(entry="run", topologies=(Topology("cta", 8),))
class CanonicalMeshes:
    @func
    def scale(
        x: Tensor[(8, 8), "bf16", "rmem"]
    ) -> Tensor[(8, 8), "bf16", "rmem"]:
        with cta_2 as _cta_2:
            v0 = mul(x, x)
            return v0

    @func
    def run(
        x: Tensor[(8, 8), "bf16", (8, 8 @ cta.b), "rmem"]
    ) -> Tensor[(8, 8), "bf16", (8, 8 @ cta.b), "rmem"]:
        with cta as _cta:
            v0 = scale(x)
            return v0
