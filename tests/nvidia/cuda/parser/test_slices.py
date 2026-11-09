"""Parser diagnostics for authored tile-window slices."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.core._corpus import Case, case, check_case, reload, special
from tilefoundry import inspection, services
from tilefoundry.dsl import *
from tilefoundry.ir import core, expr
from tilefoundry.ir.hir import tensor

TILE_WINDOW_SYNTAX = Path(__file__).parents[1] / "fixtures" / "shapes" / "tile_window_syntax.py"


def _loops(owner) -> dict[str, core.LoopRegion]:
    return {
        loop.induction_var.name: loop
        for loop in core.collect_exprs(owner.entry_function().body)
        if isinstance(loop, core.LoopRegion)
    }


def _scaled_start(sliced: expr.Call, induction: expr.Var) -> tuple[expr.Call, int]:
    raw_start = sliced.args[1].elements[0]
    scaled = [
        node
        for node in core.collect_exprs(raw_start)
        if isinstance(node, expr.Call)
        and isinstance(node.target, expr.DimMul)
        and induction in node.args
    ]
    assert len(scaled) == 1
    start = scaled[0]
    offset = expr.static_dim_value(services.normalize_dim(services.simplify_dim("sub", (raw_start, start))))
    assert any(expr.static_dim_value(arg) == 2 for arg in start.args)
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
            node
            for node in core.collect_exprs(inner.body)
            if isinstance(node, expr.Call)
            and isinstance(node.target, tensor.Slice)
            and node.target.strides == (2, 1)
        ]
        assert len(slices) == 4
        offsets: dict[tuple[int, int], expr.Call] = {}
        for sliced in slices:
            assert sliced.target.sizes == (2, 2)
            assert sliced.target.strides == (2, 1)
            start, outer_offset = _scaled_start(sliced, outer_param)
            assert isinstance(start.target, expr.DimMul)
            inner_offset = expr.static_dim_value(
                services.normalize_dim(
                    services.simplify_dim(
                        "sub",
                        (sliced.args[1].elements[1], inner.induction_var),
                    )
                )
            )
            assert inner_offset is not None
            offsets[outer_offset, inner_offset] = sliced
        assert set(offsets) == {(0, 0), (0, -1), (1, 0), (-1, 0)}

        printed = inspection.as_script(state)
        for index in (
            "x[m * 2, n]",
            "x[m * 2, n - 1]",
            "x[m * 2 + 1, n]",
            "x[m * 2 - 1, n]",
        ):
            assert index in printed
        restored = reload(printed, "NestedScaledTileWindows")
        assert inspection.as_script(restored) == printed
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
            node
            for node in core.collect_exprs(state.entry_function().body)
            if isinstance(node, core.MeshRegion)
        )
        mesh_param = next(
            param for param, argument in mesh.captures() if argument is loop.induction_var
        )
        sliced = next(
            node
            for node in core.collect_exprs(mesh.body)
            if isinstance(node, expr.Call) and isinstance(node.target, tensor.Slice)
        )
        assert sliced.target.sizes == (2, 8)
        assert sliced.target.strides == (2, 1)
        _start, offset = _scaled_start(sliced, mesh_param)
        assert offset == 1

        printed = inspection.as_script(state)
        assert "x[m * 2 + 1, :]" in printed
        restored = reload(printed, "MeshInsideTileWindow")
        assert inspection.as_script(restored) == printed
        restored_loop = _loops(restored)["m"]
        restored_mesh = next(
            node
            for node in core.collect_exprs(restored.entry_function().body)
            if isinstance(node, core.MeshRegion)
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
