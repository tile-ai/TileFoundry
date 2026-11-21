"""One loop per way a loop index's bounds decide whether it is the same on every GPU.

DirectCoord and HiddenCoord pick at the GPU's own coordinate instead, read
directly or returned by a function, so the row differs per GPU. IndexArgument
passes the index to a plain parameter of a callee; CoordArgument passes the
coordinate, which a plain parameter refuses.

Each loop picks a row of a Broadcast table at the index. On the GPU target gmem
is held per GPU, so the picked row shows how the index reads.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu, cta = Topology("gpu", 2), Topology("cta", 1)
N = DimVar("N", 1, 8)


@module(entry="forward", target=target, topologies=(gpu, cta))
class LiteralBounds:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def forward(x: Tensor[(8, 8), "f32", "gmem"], acc0: Tensor[(1, 8), "f32", "gmem"]):
        with mesh:
            acc = acc0
            for t in range(4):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class PlainCount:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def forward(x: Tensor[(8, 8), "f32", "gmem"], acc0: Tensor[(1, 8), "f32", "gmem"], n: Tensor[(), "i64", "gmem"]):
        with mesh:
            acc = acc0
            for t in range(n):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class DimVarExtent:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def forward(x: Tensor[(8, 8), "f32", "gmem"], acc0: Tensor[(1, 8), "f32", "gmem"]):
        with mesh:
            acc = acc0
            for t in range(N):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class OuterIndexStart:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def forward(x: Tensor[(8, 8), "f32", "gmem"], acc0: Tensor[(1, 8), "f32", "gmem"]):
        with mesh:
            acc = acc0
            for i in range(2):
                for t in range(i + 1, 4):
                    acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class CoordStart:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def forward(x: Tensor[(8, 8), "f32", "gmem"], acc0: Tensor[(1, 8), "f32", (1, 8) @ {mesh.g @ V()}, "gmem"]):
        with mesh:
            acc = acc0
            for t in range(mesh.g, 4):
                acc = acc + tf.index_select(x, tf.reshape(t, new_shape=(1,)), dim=0)
            return acc


@module(entry="forward", target=target, topologies=(gpu, cta))
class DirectCoord:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def forward(x: Tensor[(8, 8), "f32", "gmem"]):
        with mesh:
            return tf.index_select(x, tf.reshape(mesh.g, new_shape=(1,)), dim=0)


@module(entry="forward", target=target, topologies=(gpu, cta))
class HiddenCoord:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def rank():
        with Mesh(("gpu",), layout=(2,), names=("g",)) as m:
            return tf.reshape(m.g, new_shape=(1,))

    @func
    def forward(x: Tensor[(8, 8), "f32", "gmem"]):
        with mesh:
            return tf.index_select(x, rank(), dim=0)


@module(entry="forward", target=target, topologies=(gpu, cta))
class IndexArgument:
    mesh = Mesh(("gpu",), (2,), names=("g",))

    @func
    def pick(x: Tensor[(8, 8), "f32", "gmem"], i: Tensor[(1,), "i64", "umat"]):
        with mesh:
            return tf.index_select(x, i, dim=0)

    @func
    def forward(x: Tensor[(8, 8), "f32", "gmem"], acc0: Tensor[(1, 8), "f32", "gmem"]):
        with mesh:
            acc = acc0
            for t in range(4):
                acc = acc + pick(x, tf.reshape(t, new_shape=(1,)))
            return acc
