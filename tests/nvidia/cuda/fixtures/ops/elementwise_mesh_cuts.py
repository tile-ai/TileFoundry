"""``ops::elementwise`` over the three ways a mesh can name what holds a tile.

A stride-0 operand is the whole of broadcast. Beyond it, a tile can be cut by
two levels at once -- a CTA takes a block of it and a thread takes a row of
that -- or by two axes of one level, which is the same cut written as a grid.
"""

from __future__ import annotations

from tests.core.fixtures.tir.layouts import bcast
from tilefoundry.dsl import *
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.target import CpuTarget
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")

_CTAS, _THREADS = 4, 8
_ROWS = _CTAS * _THREADS


@module(topologies=(Topology("thread", 32),))
class ColumnBroadcast:
    """Multiply by a stride-0 column and subtract the tile from that column."""

    @prim_func(target=_CUDA)
    def col_bcast_device(
        a: Tensor[(32,), "f32"],
        r: Tensor[(4,), "f32"],
        out: Tensor[(32,), "f32"],
        out_sub: Tensor[(32,), "f32"],
        out_scalar: Tensor[(32,), "f32"],
    ):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as m:
            a_view = T.tensor_view(T.ptr_of(a), shape=(4, 8), layout=bcast((4, 8), (8, 1), m))
            r_view = T.tensor_view(T.ptr_of(r), shape=(4, 1), layout=bcast((4, 1), (1, 1), m))
            out_view = T.tensor_view(T.ptr_of(out), shape=(4, 8), layout=bcast((4, 8), (8, 1), m))
            out_sub_view = T.tensor_view(
                T.ptr_of(out_sub), shape=(4, 8), layout=bcast((4, 8), (8, 1), m)
            )
            out_scalar_view = T.tensor_view(
                T.ptr_of(out_scalar), shape=(4, 8), layout=bcast((4, 8), (8, 1), m)
            )
            lhs = T.alloc_tensor(Tensor[(4, 8), "f32", bcast((4, 8), (8, 1), m), "rmem"])
            col = T.alloc_tensor(Tensor[(4, 1), "f32", bcast((4, 1), (1, 1), m), "rmem"])
            dst = T.alloc_tensor(Tensor[(4, 8), "f32", bcast((4, 8), (8, 1), m), "rmem"])
            dst_sub = T.alloc_tensor(Tensor[(4, 8), "f32", bcast((4, 8), (8, 1), m), "rmem"])
            scalar = T.alloc_tensor(Tensor[(), "f32", Layout((), ()), "rmem"])
            dst_scalar = T.alloc_tensor(Tensor[(4, 8), "f32", bcast((4, 8), (8, 1), m), "rmem"])
            T.copy(a_view, lhs)
            T.copy(r_view, col)
            T.binary(lhs, col, dst, kind=BinaryKind.MUL)
            T.copy(dst, out_view)
            T.binary(col, lhs, dst_sub, kind=BinaryKind.SUB)
            T.copy(dst_sub, out_sub_view)
            T.fill(scalar, 0.5)
            T.binary(lhs, scalar, dst_scalar, kind=BinaryKind.ADD)
            T.copy(dst_scalar, out_scalar_view)


@module(topologies=(Topology("cta", _CTAS), Topology("thread", _THREADS)))
class TwoLevels:
    """One tile cut by two levels: a CTA takes a block, a thread takes a row."""

    @prim_func(target=_CUDA)
    def two_levels_device(a: Tensor[(_ROWS,), "f32"], out: Tensor[(_ROWS,), "f32"]):
        with Mesh(
            (Topology("cta", _CTAS), Topology("thread", _THREADS)),
            Layout(shape=(_CTAS, _THREADS), strides=(_THREADS, 1)),
            ("c", "t"),
        ) as m:
            row = ShardLayout(Layout((_ROWS,), (1,)), (Split(0), Split(0)), m)
            source = T.tensor_view(T.ptr_of(a), layout=row)
            written = T.tensor_view(T.ptr_of(out), layout=row)
            held = T.alloc_tensor(Tensor[(_ROWS,), "f32", row, "rmem"])
            T.copy(source, held)
            T.binary(held, held, held, kind=BinaryKind.MUL)
            T.copy(held, written)


@module(topologies=(Topology("thread", _ROWS),))
class TwoAxes:
    """The same cut written as one level's grid: two axes over one tensor axis."""

    @prim_func(target=_CUDA)
    def two_axes_device(a: Tensor[(_ROWS,), "f32"], out: Tensor[(_ROWS,), "f32"]):
        with Mesh(
            (Topology("thread", _ROWS),),
            Layout(shape=(_CTAS, _THREADS), strides=(_THREADS, 1)),
            ("w", "t"),
        ) as m:
            row = ShardLayout(Layout((_ROWS,), (1,)), (Split(0), Split(0)), m)
            source = T.tensor_view(T.ptr_of(a), layout=row)
            written = T.tensor_view(T.ptr_of(out), layout=row)
            held = T.alloc_tensor(Tensor[(_ROWS,), "f32", row, "rmem"])
            T.copy(source, held)
            T.binary(held, held, held, kind=BinaryKind.MUL)
            T.copy(held, written)


@module(entry="elementwise_host", target=_CUDA)
class Elementwise:
    """One host entry over the three ways a mesh names what holds a tile."""

    bcast_tier = ColumnBroadcast
    two_levels = TwoLevels
    two_axes = TwoAxes

    @prim_func(target=CpuTarget())
    def elementwise_host(
        a: Tensor[(32,), "f32"],
        r: Tensor[(4,), "f32"],
        out: Tensor[(32,), "f32"],
        out_sub: Tensor[(32,), "f32"],
        out_scalar: Tensor[(32,), "f32"],
        level_a: Tensor[(_ROWS,), "f32"],
        level_out: Tensor[(_ROWS,), "f32"],
        axis_a: Tensor[(_ROWS,), "f32"],
        axis_out: Tensor[(_ROWS,), "f32"],
    ):
        launch(  # noqa: F821
            bcast_tier.col_bcast_device,
            a,
            r,
            out,
            out_sub,
            out_scalar,
            grid=(1, 1, 1),
            block=(32, 1, 1),  # noqa: F821
        )
        launch(  # noqa: F821
            two_levels.two_levels_device,  # noqa: F821
            level_a,
            level_out,
            grid=(_CTAS, 1, 1),
            block=(_THREADS, 1, 1),
        )
        launch(  # noqa: F821
            two_axes.two_axes_device,  # noqa: F821
            axis_a,
            axis_out,
            grid=(1, 1, 1),
            block=(_ROWS, 1, 1),
        )
