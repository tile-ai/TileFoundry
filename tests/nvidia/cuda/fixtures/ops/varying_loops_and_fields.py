"""Values a loop or a tuple field makes Varying, or leaves Broadcast.

``TripCount``: the loop runs a different number of times on each GPU, so its
result is Varying although each iteration only adds the same ``x``.
``OutsideBound``: the bound is Varying on the CTA topology, which the thread
scope of the loop does not name; the index and the result are still Varying there.
``TupleField``: reading the Broadcast field of a tuple whose other field is
Varying gives a Broadcast value.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu, cta = Topology("gpu", 2), Topology("cta", 1)


@module(entry="forward", target=target, topologies=(gpu, cta))
class TripCount:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def forward(x: Tensor[(8,), "f32", "rmem"]):
        with mesh:
            n = mesh.g + 1
            acc = x
            for _i in range(n):
                acc = acc + x
            return acc


@module(entry="forward", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 4), Topology("thread", 32)))
class OutsideBound:
    ctas = Mesh(("cta",), (4,), names=("x",))
    threads = Mesh(("thread",), (32,), names=("t",))

    @func
    def forward(n: Tensor[(1,), "i64", (1,) @ {ctas.x @ V()}], x: Tensor[(8,), "f32", "rmem"]):
        with threads:
            acc = x
            for _i in range(n[0]):
                acc = acc + x
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class TupleField:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def pair(x: Tensor[(8,), "f32", "rmem"]):
        with mesh:
            v = x * tf.cast(mesh.g, "f32")
            return v, x

    @func
    def forward(x: Tensor[(8,), "f32", "rmem"]):
        with mesh:
            v, b = pair(x)
            return b
