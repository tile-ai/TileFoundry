"""A Varying value where the type says Broadcast: an argument, a loop carry, a schedule target.

Each GPU's coordinate differs, so a value read from it is Varying; a plain
parameter or buffer is Broadcast, and each must be written Varying instead.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import cuda

target = cuda.Target("nvidia.h200_sxm", device_count=2)
gpu, cta, thread = Topology("gpu", 2), Topology("cta", 1), Topology("thread", 32)


@module(entry="forward", target=target, topologies=(gpu, cta))
class CoordArgument:
    @func(mesh=Mesh(("gpu",), (2,), names=("g",)))
    def pick(x: Tensor[(8, 8), "f32", "gmem"], i: Tensor[(1,), "i64", (1,), "rmem"]):
        return tf.index_select(x, i, dim=0)

    @func(mesh=Mesh(("gpu",), (2,), names=("g",)))
    def forward(x: Tensor[(8, 8), "f32", "gmem"]):
        return pick(x, tf.reshape(mesh.g, new_shape=(1,)))


@module(entry="forward", target=target, topologies=(gpu, cta))
class LoopCarry:
    @func(mesh=Mesh(("gpu",), (2,), names=("g",)))
    def forward(x: Tensor[(8, 8), "f32", "gmem"], acc0: Tensor[(1, 8), "f32", "gmem"]):
        acc = acc0
        for _t in range(4):
            acc = acc + tf.index_select(x, tf.reshape(mesh.g, new_shape=(1,)), dim=0)
        return acc


@module(entry="run", target=cuda.Target("nvidia.h200_sxm"), topologies=(cta, thread))
class ScheduleTarget:
    @func
    def run(x: Tensor[(32,), "i64"]) -> Tensor[(32,), "i64"]:
        with Mesh(("cta",), layout=(1,), names=("block",)) as _cta:
            with Mesh(("thread",), layout=(32,), names=("lane",)) as lanes:
                lane = tf.reshape(lanes.lane, new_shape=(1,))
                value_dst = tf.empty(Tensor[(1,), "i64", (1,), "rmem"])
                value = tf.schedule((lane, value_dst), op=T.copy())
                out = tf.zeros(Tensor[(32,), "i64", None, "gmem"])
                return tf.insert_slice(out, value, (0,))
