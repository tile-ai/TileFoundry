from __future__ import annotations

from tilefoundry.dsl import *


@func
def f(a: Tensor[(1, 1536), "f32"]) -> Tensor[(1, 1536), "f32"]:
    sl = ShardLayout(
        layout=Layout((1, 1536), (1536, 1)),
        attrs=(),
        mesh=Mesh((Topology("cta", 128),), Layout((128,), (1,))),
    )
    b = tf.reshard(a, layout=sl)
    return b
