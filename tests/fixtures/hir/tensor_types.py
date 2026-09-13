"""Small HIR module covering the printer's type and region surfaces."""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget


@module(
    entry="loop_nest",
    target=CudaTarget("nvidia.h200_sxm"),
    topologies=(Topology("cta", 1), Topology("thread", 4)),
)
class TensorTypes:
    @func
    def broadcast(x: Tensor[(8,), "f32"]):
        with Mesh(("thread",), (4,), ("lane",)) as _m:
            return tf.reshard(x, (8,), "gmem")

    @func
    def split_1d(x: Tensor[(16,), "f32"]):
        with Mesh(("thread",), (4,), ("lane",)) as m:
            local = tf.reshard(x, (16 @ m.lane,), "rmem")
            return tf.reshard(local, (16,), "gmem")

    @func
    def split_2d(x: Tensor[(8, 16), "f32"]):
        with Mesh(("thread",), (4,), ("lane",)) as m:
            local = tf.reshard(x, (8, 16 @ m.lane), "rmem")
            return tf.reshard(local, (8, 16), "gmem")

    @func
    def partial(x: Tensor[(8,), "f32"]):
        with Mesh(("thread",), (4,), ("lane",)) as m:
            local = tf.reshard(x, (8 @ m.lane,), "rmem")
            return tf.reduce(local, (-1,), True, ReduceKind.SUM)

    @func
    def mixed(x: Tensor[(8, 16), "f32"]):
        with Mesh(("thread", "cta"), (4, 1), ("lane", "tile")) as m:
            local = tf.reshard(x, (8 @ m.lane, 16 @ m.tile), "rmem")
            return tf.reshard(local, (8, 16), "gmem")

    @func
    def nested_mesh(x: Tensor[(8,), "f32"]):
        with Mesh(("thread",), (4,), ("outer",)) as _outer:
            with Mesh(("thread",), (4,), ("inner",)) as inner:
                local = tf.reshard(x, (8 @ inner.inner,), "rmem")
                return tf.reshard(local, (8,), "gmem")

    @func
    def loop_nest(x: Tensor[(8,), "f32"]):
        with Mesh(("thread",), (4,), ("lane",)) as m:
            carried = tf.reshard(x, (8 @ m.lane,), "rmem")
            for _ in range(2):
                carried = tf.square(carried)
            return tf.reshard(carried, (8,), "gmem")
