"""Parser diagnostics for authored tile-window slices."""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tests.fixtures.shapes.tile_window_syntax import (
    MeshInsideTileWindow,
    NestedScaledTileWindows,
)
from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.isl_interop import normalize_dim
from tilefoundry.ir.types.dim import DimMul, DimSub, simplify_dim
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.parser import ParseError

N = DimVar("N", 1, 256)


def test_a_tile_window_cannot_be_used_as_an_explicit_bound() -> None:
    with pytest.raises(
        ParseError,
        match=r"tile loop variable is already a window.*x\[:, t, :\].*base = t \+ 0",
    ):

        @func
        def stage(x: Tensor[(1, 256, 64), "f32"]):
            out = x[:, 0:128, :]
            for t in tile(128, 128):
                out = x[:, t : t + 128, :]
            return out


def test_a_tile_window_rejects_true_division() -> None:
    with pytest.raises(ParseError, match=r"tile windows only support ± c and \* c"):

        @func
        def stage(x: Tensor[(1, 256, 64), "f32"]):
            out = x[:, 0:64, :]
            for t in tile(128, 128):
                out = x[:, t / 2, :]
            return out


def test_a_literal_index_rejects_true_division() -> None:
    with pytest.raises(ParseError, match=r"index arithmetic uses //"):

        @func
        def stage(x: Tensor[(1, 256, 64), "f32"]):
            return x[:, 8 / 2, :]


def test_a_symbolic_index_rejects_true_division() -> None:
    with pytest.raises(ParseError, match=r"index arithmetic uses //"):

        @func
        def stage(x: Tensor[(1, 256, 64), "f32"]):
            return x[:, N / 2, :]


def _loops(owner) -> dict[str, LoopRegion]:
    return {
        loop.induction_var.name: loop
        for loop in collect_exprs(owner.entry_function().body)
        if isinstance(loop, LoopRegion)
    }


def _scaled_start(sliced: Call, induction: Var) -> tuple[Call, int]:
    raw_start = sliced.args[1].elements[0]
    scaled = [
        expr
        for expr in collect_exprs(raw_start)
        if isinstance(expr, Call)
        and isinstance(expr.target, DimMul)
        and induction in expr.args
    ]
    assert len(scaled) == 1
    start = scaled[0]
    offset = static_dim_value(normalize_dim(simplify_dim(DimSub, (raw_start, start))))
    assert any(static_dim_value(arg) == 2 for arg in start.args)
    return start, offset


def test_nested_tile_windows_capture_and_transform_the_outer_window() -> None:
    loops = _loops(NestedScaledTileWindows)
    outer, inner = loops["m"], loops["n"]
    outer_param = next(
        param for param, argument in inner.captures() if argument is outer.induction_var
    )
    slices = [
        expr
        for expr in collect_exprs(NestedScaledTileWindows.entry_function().body)
        if isinstance(expr, Call)
        and isinstance(expr.target, Slice)
        and expr.target.strides == (2, 1)
    ]
    assert len(slices) == 4
    offsets: dict[tuple[int, int], Call] = {}
    for sliced in slices:
        assert sliced.target.sizes == (2, 2)
        assert sliced.target.strides == (2, 1)
        start, outer_offset = _scaled_start(sliced, outer_param)
        assert isinstance(start.target, DimMul)
        inner_offset = static_dim_value(
            normalize_dim(
                simplify_dim(
                    DimSub,
                    (sliced.args[1].elements[1], inner.induction_var),
                )
            )
        )
        assert inner_offset is not None
        offsets[outer_offset, inner_offset] = sliced
    assert set(offsets) == {(0, 0), (0, -1), (1, 0), (-1, 0)}

    printed = as_script(NestedScaledTileWindows)
    for index in (
        "x[m * 2, n]",
        "x[m * 2, n - 1]",
        "x[m * 2 + 1, n]",
        "x[m * 2 - 1, n]",
    ):
        assert index in printed
    restored = import_dsl(printed, "NestedScaledTileWindows")
    assert as_script(restored) == printed
    assert {
        name: (len(loop.params), len(loop.args)) for name, loop in _loops(restored).items()
    } == {name: (len(loop.params), len(loop.args)) for name, loop in loops.items()}


def test_a_mesh_inside_a_tile_loop_captures_the_outer_window() -> None:
    loop = _loops(MeshInsideTileWindow)["m"]
    mesh = next(
        expr
        for expr in collect_exprs(MeshInsideTileWindow.entry_function().body)
        if isinstance(expr, MeshRegion)
    )
    mesh_param = next(
        param for param, argument in mesh.captures() if argument is loop.induction_var
    )
    sliced = next(
        expr
        for expr in collect_exprs(mesh.body)
        if isinstance(expr, Call) and isinstance(expr.target, Slice)
    )
    assert sliced.target.sizes == (2, 8)
    assert sliced.target.strides == (2, 1)
    _start, offset = _scaled_start(sliced, mesh_param)
    assert offset == 1

    printed = as_script(MeshInsideTileWindow)
    assert "x[m * 2 + 1, :]" in printed
    restored = import_dsl(printed, "MeshInsideTileWindow")
    assert as_script(restored) == printed
    restored_loop = _loops(restored)["m"]
    restored_mesh = next(
        expr
        for expr in collect_exprs(restored.entry_function().body)
        if isinstance(expr, MeshRegion)
    )
    assert (len(restored_loop.params), len(restored_loop.args)) == (
        len(loop.params),
        len(loop.args),
    )
    assert (len(restored_mesh.params), len(restored_mesh.args)) == (
        len(mesh.params),
        len(mesh.args),
    )
