"""Values made from Varying operands whose storage is one copy for many units (D37).

``SharedGather``: each thread gathers its own row of a shared-memory table; the rows
are a new value, so they are Varying although shared memory is one copy per CTA.
``SharedStore``: each CTA inserts its own rows into one global output; a write into
a destination keeps the one-copy rule, so the output stays Broadcast.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

_H200 = cuda.Target("nvidia.h200_sxm")


@module(entry="forward", target=_H200, topologies=(Topology("cta", 1), Topology("thread", 4)))
class SharedGather:
    threads = Mesh(("thread",), (4,), names=("t",))

    @func
    def forward(table: Tensor[(4, 3), "f32", "gmem"]):
        with Mesh(("cta",), layout=(1,), names=("b",)):
            with threads:
                shared = tf.reshard(table, (4, 3), "smem")
                rows = tf.index_select(shared, tf.reshape(threads.t, new_shape=(1,)), dim=0)
                return tf.reshard(rows, (1, 3) @ {threads.t @ V()}, "gmem")


@module(entry="forward", target=_H200, topologies=(Topology("cta", 4), Topology("thread", 32)))
class SharedStore:
    ctas = Mesh(("cta",), (4,), names=("c",))

    @func
    def forward(x: Tensor[(4, 8), "f32", "gmem"]) -> Tensor[(4, 8), "f32", (4, 8), "gmem"]:
        with ctas:
            out = tf.zeros(Tensor[(4, 8), "f32"])
            return tf.insert_slice(out, x[ctas.c:ctas.c + 1, :], (ctas.c, 0))
