"""Placed values covering every shape the printer's layout sugar can take.

One entry per combination the sugar has to survive: a mesh level alone and two
levels composed on one value, splits on one and on several tensor axes, every
value state a mesh axis can hold, contiguous and explicitly strided layouts
over the same logical shape, and a loop whose carried fields are placed
differently from each other.
"""

from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import B, Layout, Mesh, P, Topology
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 4), Topology("thread", 8))

_TILE = Mesh((Topology("cta", 4),), Layout((4,), (1,)), names=("tile",))
_WARP_LANE = Mesh((Topology("thread", 8),), Layout((2, 4), (4, 1)), names=("warp", "lane"))
_LANES = Mesh((Topology("thread", 8),), Layout((8,), (1,)), names=("lane",))


@module(entry="composed_mesh_pipeline", target=_H200, topologies=_TOPOLOGIES)
class TypePrinterSugar:
    @func
    def composed_mesh_pipeline(
        x: Tensor[(8, 4, 16), "f32"],
        acc: Tensor[
            (8, 16), "f32",
            ((2 @ _WARP_LANE.warp, 4, 16), {_WARP_LANE.lane @ P("sum")}),
            "rmem",
        ],
        mixed: Tensor[
            (8, 16), "f32",
            ((8 @ _TILE.tile, 16), {_WARP_LANE.warp @ B(), _WARP_LANE.lane @ P("sum")}),
            "rmem",
        ],
    ):
        with _TILE as cta:
            with _WARP_LANE as thr:
                composed = tf.reshard(
                    x, (8 @ cta.tile, 4 @ thr.warp, 16 @ thr.lane), "rmem"
                )
                staged = tf.reshard(tf.square(composed), (8 @ cta.tile, 4, 16), "smem")
                narrowed = tf.cast(staged, dtype="bf16")
                swapped = tf.transpose(narrowed, perm=(0, 2, 1))
                gathered = tf.reshard(swapped, (8, 16, 4), "gmem")
                folded = tf.reshard(acc, ((8 @ thr.warp, 16 @ thr.lane)), "rmem")
                summed = tf.reshard(
                    mixed, ((8, 16), {cta.tile @ B(), thr.warp @ B(), thr.lane @ B()}), "rmem"
                )
                for _ in range(3):
                    folded = tf.square(folded)
                    summed = tf.add(summed, summed)
                return (
                    gathered,
                    tf.reshard(folded, (8, 16), "gmem"),
                    tf.reshard(summed, (8, 16), "gmem"),
                )

    @func
    def nested_loop_tuple(
        x: Tensor[(8, 16), "f32"],
        weight: Tensor[
            (8, 16), "f32", ((8, 16), {_WARP_LANE.warp @ B(), _WARP_LANE.lane @ P("max")})
        ],
    ):
        with _LANES as lanes:
            split = tf.reshard(x, (8 @ lanes.lane, 16), "rmem")
            whole = tf.reshard(x, ((8, 16), {lanes.lane @ B()}), "rmem")
            for _ in range(3):
                split = tf.square(split)
                whole = tf.add(whole, whole)
            with _WARP_LANE as thr:
                unfolded = tf.reshard(
                    weight, ((8, 16), {thr.warp @ B(), thr.lane @ B()}), "gmem"
                )
            return (
                tf.reshard(split, (8, 16), "gmem"),
                tf.reshard(whole, (8, 16), "gmem"),
                unfolded,
            )
