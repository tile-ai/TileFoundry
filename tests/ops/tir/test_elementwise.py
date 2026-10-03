"""``ops::elementwise`` over the three ways a mesh can name what holds a tile.

A stride-0 operand is the whole of broadcast. Beyond it, a tile can be cut by
two levels at once -- a CTA takes a block of it and a thread takes a row of
that -- or by two axes of one level, which is the same cut written as a grid.

See [runtime §2.6](docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
import tilefoundry.codegen.cuda  # noqa: F401 -- trigger emitter autodiscovery
from tests.fixtures.tir.layouts import bcast
from tilefoundry import module, prim_func
from tilefoundry.dsl import T, Tensor
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.tir.arith import Binary
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Return, Sequential
from tilefoundry.ir.types import Layout, Mesh, ShardLayout, Split, Topology, make_tensor_type
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.visitor_registry.verify import verify_prim_function

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
    ):
        with Mesh((Topology("thread", 32),), Layout(shape=(32,), strides=(1,)), ("t",)) as m:
            a_view = T.tensor_view(T.ptr_of(a), layout=bcast((32,), (1,), m))
            r_view = T.tensor_view(T.ptr_of(r), layout=bcast((4,), (1,), m))
            out_view = T.tensor_view(T.ptr_of(out), layout=bcast((32,), (1,), m))
            out_sub_view = T.tensor_view(T.ptr_of(out_sub), layout=bcast((32,), (1,), m))
            lhs = T.alloc_tensor(Tensor[(4, 8), "f32", bcast((4, 8), (8, 1), m), "rmem"])
            col = T.alloc_tensor(Tensor[(4, 1), "f32", bcast((4, 1), (1, 1), m), "rmem"])
            dst = T.alloc_tensor(Tensor[(4, 8), "f32", bcast((4, 8), (8, 1), m), "rmem"])
            dst_sub = T.alloc_tensor(Tensor[(4, 8), "f32", bcast((4, 8), (8, 1), m), "rmem"])
            T.copy(a_view, lhs)
            T.copy(r_view, col)
            T.binary(lhs, col, dst, kind=BinaryKind.MUL)
            T.copy(dst, out_view)
            T.binary(col, lhs, dst_sub, kind=BinaryKind.SUB)
            T.copy(dst_sub, out_sub_view)


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


def test_binary_refuses_an_operand_outside_register_memory() -> None:
    types = (
        make_tensor_type((8,), storage="smem"),
        make_tensor_type((8,), storage="rmem"),
        make_tensor_type((8,), storage="rmem"),
    )
    args = tuple(Var(type=type_, name=f"a{index}") for index, type_ in enumerate(types))
    fn = PrimFunction(
        name="fn",
        params=args,
        body=Sequential(
            body=(
                Evaluate(callable=Binary(kind=BinaryKind.ADD), args=args),
                Return(),
            )
        ),
    )
    with pytest.raises(VerifyError, match=r"lhs .* does not match StorageKind.RMEM"):
        verify_prim_function(fn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_mesh_names_what_holds_a_tile_three_ways() -> None:
    """Broadcast, two levels over one tile, and two axes over one tensor axis.

    The last two cut the same tile the same way and must agree: a CTA taking a
    block of rows while a thread takes one of them is the grid one level writes
    as two axes.
    """
    rm = tilefoundry.compile(Elementwise, target=_CUDA)
    torch.manual_seed(0)
    a = torch.randn(32, dtype=torch.float32, device="cuda")
    r = torch.randn(4, dtype=torch.float32, device="cuda")
    out = torch.zeros(32, dtype=torch.float32, device="cuda")
    out_sub = torch.zeros_like(out)
    rows = torch.randn(_ROWS, dtype=torch.float32, device="cuda")
    level_out = torch.zeros(_ROWS, dtype=torch.float32, device="cuda")
    axis_out = torch.zeros(_ROWS, dtype=torch.float32, device="cuda")

    rm(a, r, out, out_sub, rows, level_out, rows, axis_out)
    torch.cuda.synchronize()

    assert torch.allclose(out, a * r.repeat(8), rtol=0, atol=0)
    assert torch.allclose(out_sub, r.repeat(8) - a, rtol=0, atol=0)
    assert torch.allclose(level_out, rows * rows, rtol=0, atol=0)
    assert torch.equal(level_out, axis_out)
