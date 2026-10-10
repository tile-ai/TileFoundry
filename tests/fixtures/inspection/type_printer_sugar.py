"""Placed values covering every shape the printer's layout sugar can take.

One entry per combination the sugar has to survive: a mesh level alone and two
levels composed on one value, splits on one and on several tensor axes, every
value state a mesh axis can hold, contiguous and explicitly strided layouts
over the same logical shape, and a loop whose carried fields are placed
differently. The same fragment arrangement is referenced by two entries:
one binds its mesh for layout sugar, while the other keeps it in a signature.
"""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.target import CudaTarget

_H200 = CudaTarget("nvidia.h200_sxm")
_TOPOLOGIES = (Topology("cta", 4), Topology("thread", 8))


@module(entry="composed_mesh_pipeline", target=_H200, topologies=_TOPOLOGIES)
class TypePrinterSugar:
    @func(mesh=Mesh((Topology('thread', 8),), Layout((2, 4), (4, 1)), names=('warp', 'lane')))
    def composed_mesh_pipeline(
        x: Tensor[(8, 4, 16), "f32"],
        seed: Tensor[(16,), "f32"],
        acc: Tensor[
            (8, 16), "f32",
            ((2 @ mesh.warp, 4, 16), {mesh.lane @ P("sum")}),
            "rmem",
        ],
        mixed: Tensor[(8, 16), "f32", ShardLayout(layout=Layout((4, 2, 16), None), attrs=(S(0), B(), P('sum')), mesh=Mesh((Topology('cta', 4), Topology('thread', 8)), Layout((4, 2, 4), (8, 4, 1)), names=('tile', 'warp', 'lane'))), "rmem"],
    ):
        with Mesh(("cta",), layout=(4,), names=("tile",)) as cta:
            composed = tf.reshard(
                x, (8 @ cta.tile, 4 @ mesh.warp, 16 @ mesh.lane), "rmem"
            )
            staged = tf.reshard(tf.square(composed), (8 @ cta.tile, 4, 16), "smem")
            narrowed = tf.cast(staged, dtype="bf16")
            swapped = tf.transpose(narrowed, perm=(0, 2, 1))
            gathered = tf.reshard(swapped, (8, 16, 4), "gmem")
            seeded = tf.reshard(
                seed, ShardLayout(Layout((2, 4, 2), (8, 2, 1)), (S(0), S(1)), mesh), "rmem"
            )
        folded = tf.reshard(acc, (8 @ mesh.warp, 16 @ mesh.lane), "rmem")
        summed = tf.reshard(
            mixed, ((8, 16), {mesh.warp @ B(), mesh.lane @ B()}), "rmem"
        )
        for _ in range(3):
            folded = tf.square(folded)
            summed = tf.add(summed, summed)
        return (
            gathered,
            tf.reshard(folded, (8, 16), "gmem"),
            tf.reshard(summed, (8, 16), "gmem"),
            tf.reshard(seeded, (16,), "gmem"),
        )

    @func
    def nested_loop_tuple(
        x: Tensor[(8, 16), "f32"],
        weight: Tensor[(8, 16), "f32", ShardLayout(layout=Layout((8, 16), None), attrs=(B(), P('max')), mesh=Mesh((Topology('thread', 8),), Layout((2, 4), (4, 1)), names=('warp', 'lane')))],
    ):
        with Mesh(("thread",), layout=(8,), names=("lane",)) as lanes:
            split = tf.reshard(x, (8 @ lanes.lane, 16), "rmem")
            whole = tf.reshard(x, ((8, 16), {lanes.lane @ B()}), "rmem")
            for _ in range(3):
                split = tf.square(split)
                whole = tf.add(whole, whole)
            with Mesh(("thread",), layout=(2, 4), names=("warp", "lane")) as thr:
                per_warp = tf.reshard(weight, (8 @ thr.warp, 16), "rmem")
                unfolded = tf.reshard(
                    per_warp, ((8, 16), {thr.warp @ B(), thr.lane @ B()}), "gmem"
                )
            return (
                tf.reshard(split, (8, 16), "gmem"),
                tf.reshard(whole, (8, 16), "gmem"),
                unfolded,
            )

    @func(mesh=Mesh((Topology('cta', 4),), Layout((4,), (1,)), names=('tile',)))
    def named_and_out_of_scope(
        x: Tensor[(8, 16), "f32"],
        held: Tensor[(8, 16), "f32", (8 @ mesh.tile, 16), "rmem"],  # noqa: F821
        frag: Tensor[(16,), "f32", ShardLayout(layout=Layout((2, 4, 2), (8, 2, 1)), attrs=(S(0), S(1)), mesh=Mesh((Topology('thread', 8),), Layout((2, 4), (4, 1)), names=('warp', 'lane'))), "rmem"],
    ):
        """The two placements sugar cannot state.

        ``frag`` holds the same layout `composed_mesh_pipeline` reshards to,
        but this function never enters that layout's mesh, so nothing here
        binds a name sugar could use. ``escaped`` is resharded onto a mesh this
        scope never enters -- a reshard is the one operation allowed to cross
        that boundary, so its target names a mesh that is not its own scope.
        """
        mine = tf.reshard(x, (8 @ mesh.tile, 16), "rmem")  # noqa: F821
        escaped = tf.reshard(x, ShardLayout(layout=Layout((2, 4, 16), None), attrs=(S(0), B()), mesh=Mesh((Topology('thread', 8),), Layout((2, 4), (4, 1)), names=('warp', 'lane'))), "rmem")
        return tf.reshard(mine, (8, 16), "gmem"), held, frag, escaped
