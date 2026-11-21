"""Values a loop or a tuple field makes Varying or leaves Broadcast, and a lane-Varying tile.

``TripCount``: each GPU runs the loop a different number of times, so its result is
Varying. ``OutsideBound``: the bound is Varying on the CTA topology, which the loop's
thread scope does not name; the result is still Varying there. ``TupleField``: the
Broadcast field of a tuple with a Varying field is Broadcast. ``OtherSlice`` reads a
register tile Varying only on the lanes in a thread slice that holds none of its copies.
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


@module(entry="forward", target=cuda.Target("nvidia.h200_sxm"), topologies=(Topology("cta", 1), Topology("thread", 64)))
class OtherSlice:
    @func
    def forward(x: Tensor[(1, 8), "bf16", "rmem"]):
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(2, 32), names=("half", "lane")) as threads:
                with threads[0:1, :] as writer:
                    tile = tf.zeros(Tensor[(1, 8), "bf16", (1, 8) @ {writer.lane @ V()}, "rmem"])
                with threads[1:2, :] as _reader:
                    return tf.insert_slice(tile, x, (0, 0))

