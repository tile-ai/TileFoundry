"""Parser diagnostics for authored tile-window slices."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, reload, special
from tilefoundry.dsl import *
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.isl_interop import normalize_dim
from tilefoundry.ir.types.dim import DimMul, DimSub, simplify_dim
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.ir.visitor import collect_exprs

TILE_WINDOW_SYNTAX = Path(__file__).parents[1] / "fixtures" / "shapes" / "tile_window_syntax.py"


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


class SliceCase(Case):
    """A tile-window program checked by its slices and their printed form."""


@case
class NestedScaledTileWindows(SliceCase):
    path: Path = TILE_WINDOW_SYNTAX
    name: str | None = "NestedScaledTileWindows"

    def guard(self, state, problems):
        loops = _loops(state)
        outer, inner = loops["m"], loops["n"]
        outer_param = next(
            param for param, argument in inner.captures() if argument is outer.induction_var
        )
        slices = [
            expr
            for expr in collect_exprs(inner.body)
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

        printed = as_script(state)
        for index in (
            "x[m * 2, n]",
            "x[m * 2, n - 1]",
            "x[m * 2 + 1, n]",
            "x[m * 2 - 1, n]",
        ):
            assert index in printed
        restored = reload(printed, "NestedScaledTileWindows")
        assert as_script(restored) == printed
        assert {
            name: (len(loop.params), len(loop.args)) for name, loop in _loops(restored).items()
        } == {name: (len(loop.params), len(loop.args)) for name, loop in loops.items()}


@case
class MeshInsideTileWindow(SliceCase):
    path: Path = TILE_WINDOW_SYNTAX
    name: str | None = "MeshInsideTileWindow"

    def guard(self, state, problems):
        loop = _loops(state)["m"]
        mesh = next(
            expr
            for expr in collect_exprs(state.entry_function().body)
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

        printed = as_script(state)
        assert "x[m * 2 + 1, :]" in printed
        restored = reload(printed, "MeshInsideTileWindow")
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


CASES = special(SliceCase)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_slice_case(case: SliceCase) -> None:
    check_case(case)
