from __future__ import annotations

from tilefoundry.module import module
from tilefoundry import func
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.storage import gmem, host, rmem, smem, tmem  # noqa: F401
from tilefoundry.ir.types.shard import (
    B, S, P, ComposedLayout, Layout, Mesh, ShardLayout, Topology,
)

cta_2 = Mesh((Topology("cta", 8),), Layout((8,), (1,)), names=())

@module(entry="run", topologies=(Topology("cta", 8),))
class CanonicalAnonMesh:
    @func
    def run(
        x: Tensor[(8, 8), "bf16",
            ShardLayout(
                layout=Layout((8, 8), (8, 1)),
                attrs=(B(),),
                mesh=Mesh((Topology("cta", 8),), Layout((8,), (1,))),
            )]
    ) -> Tensor[(8, 8), "bf16",
        ShardLayout(
            layout=Layout((8, 8), (8, 1)),
            attrs=(B(),),
            mesh=Mesh((Topology("cta", 8),), Layout((8,), (1,))),
        )]:
        with cta_2 as _cta_2:
            v0 = mul(x, x)
            return v0
