from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget


@module(entry="f", target=CpuTarget(), topologies=(Topology("cta", 128),))
class F:
    @func
    def f(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
        sl = ShardLayout(
            layout=Layout((1, 1536), (1536, 1)),
            attrs=(),
            mesh=Mesh((Topology("cta", 128),), Layout((128,), (1,))),
        )
        b = tf.reshard(a, sl)
        return b
