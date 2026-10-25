from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry='f', target=CpuTarget(), topologies=(Topology('cta', 4), Topology('thread', 2)))
class F:
    @func
    def f(a: Tensor[(8,), 'f32', ShardLayout(ComposedLayout(None, 1, ShardLayout(Layout((8,), (1,)), (S(0),), Mesh((Topology('thread', 2),), Layout((2,), (1,))))), (B(),), Mesh((Topology('cta', 4),), Layout((4,), (1,))))]) -> Tensor[(8,), 'f32', ShardLayout(ComposedLayout(None, 1, ShardLayout(Layout((8,), (1,)), (S(0),), Mesh((Topology('thread', 2),), Layout((2,), (1,))))), (B(),), Mesh((Topology('cta', 4),), Layout((4,), (1,))))]:
        return a
