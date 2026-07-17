"""GridRegionExpr carry-out lifting tests.

``for i in tile(...)`` body Assigns whose LHS is an outer-scope Var get
lifted to ``carried_args`` + ``yield_values`` on the produced
``GridRegionExpr``.
"""

from __future__ import annotations

import textwrap

import pytest

from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.schedule.constraints import AgentConstraintsMetadata


def _dedent(src: str) -> str:
    return textwrap.dedent(src).strip()


def _find_grid(expr) -> GridRegionExpr:
    """Walk an Expr DAG and return the first GridRegionExpr found."""
    seen: set[int] = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if id(e) in seen:
            continue
        seen.add(id(e))
        if isinstance(e, GridRegionExpr):
            return e
        if isinstance(e, Call):
            stack.extend(e.args)
        if isinstance(e, GridRegionExpr):  # unreachable; for clarity
            pass
    raise AssertionError("no GridRegionExpr found in body")


# ── No carry — backward-compatible behavior --------------------------------


@func
def _no_carry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(8):
        y = relu(x)  # noqa: F841


def test_no_carry_keeps_empty_carried_args():
    fn = _no_carry
    assert isinstance(fn.body, GridRegionExpr)
    grid = fn.body
    # No-carry loop: init_args / carried_args / yield_values are all empty;
    # the loop value is driven by `body`.
    assert grid.carried_args == ()
    assert grid.init_args == ()
    assert grid.yield_values == ()


def test_tile_loop_where_assignment_parses_and_attaches_metadata():
    src = """
from tilefoundry import func
from tilefoundry.dsl import Tensor, tf

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(8):
        y: where(layout=(H @ cta,)) = tf.add(x, x)
"""
    fn = parse_script(_dedent(src))
    assert isinstance(fn.body, GridRegionExpr)
    assert isinstance(fn.body.body.metadata[0], AgentConstraintsMetadata)


# ── Iteration domain (extent / step) --------------------------------------


@func
def _two_arg_tile(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(8, 2):
        y = relu(x)  # noqa: F841


def test_single_arg_tile_step_defaults_to_one():
    grid = _no_carry.body
    assert grid.extent == 8
    assert grid.step == 1


def test_two_arg_tile_stores_extent_and_step():
    grid = _two_arg_tile.body
    assert isinstance(grid, GridRegionExpr)
    assert grid.extent == 8
    assert grid.step == 2


# ── ShapeDim (DimVar) extent / step ---------------------------------------


_SEQ = DimVar("seq_len", 1, 100)


@func
def _dimvar_extent(x: Tensor[(_SEQ, 4), "f32"]) -> Tensor[(_SEQ, 4), "f32"]:
    for i in tile(_SEQ, 2):
        y = relu(x)  # noqa: F841


def test_dimvar_extent_parses_to_shapedim():
    grid = _dimvar_extent.body
    assert isinstance(grid, GridRegionExpr)
    assert grid.extent is _SEQ
    assert grid.step == 2


TILE_NON_DIM_SRC = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(x):
        y = relu(x)
"""


def test_tile_rejects_non_dim_expr():
    # A bare tensor (not int / DimVar / dim-op Expr) is not a legal extent.
    with pytest.raises(VerifyError, match="dim expression"):
        parse_script(_dedent(TILE_NON_DIM_SRC))


# ── Single carry-out lifting ----------------------------------------------


@func
def _single_carry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    o = relu(x)
    for i in tile(8):
        o = add(o, x)
    return o


def test_single_carry_lifts_outer_var():
    fn = _single_carry
    grid = _find_grid(fn.body)
    assert len(grid.carried_args) == 1
    phi = grid.carried_args[0]
    assert isinstance(phi, Var)
    assert phi.name == "o"
    assert len(grid.yield_values) == 1
    # yield is the rebinding RHS — i.e. the `add(o, x)` Call.
    assert isinstance(grid.yield_values[0], Call)
    # GridRegionExpr.type matches the phi var's type (single carry).
    assert grid.type == phi.type
    # Outer body returns `o` which after the loop is bound to the grid.
    assert fn.body is grid


# ── Body-must-not-contain-return -------------------------------------------


RETURN_IN_BODY_SRC = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(8):
        return x
    return x
"""


def test_return_inside_tile_body_rejected():
    with pytest.raises(VerifyError, match="must not contain `return`"):
        parse_script(_dedent(RETURN_IN_BODY_SRC))


# ── AugAssign rejected (v1 only `=` is supported) --------------------------


AUGASSIGN_SRC = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    o = relu(x)
    for i in tile(8):
        o += x
    return o
"""


def test_augassign_in_body_rejected():
    with pytest.raises(VerifyError, match="augmented assignment"):
        parse_script(_dedent(AUGASSIGN_SRC))


# ── Inner-scope Assigns (non-outer LHS) are NOT carries --------------------


@func
def _inner_only(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(8):
        t = relu(x)
        z = add(t, x)  # noqa: F841


def test_inner_only_assigns_are_not_carries():
    fn = _inner_only
    assert isinstance(fn.body, GridRegionExpr)
    grid = fn.body
    # `t` and `z` are fresh inner names (not outer-scope when scanned),
    # so no carry slot is created.
    assert grid.carried_args == ()


# ── `range(...)` → GridRegion (scalar iv), unified with tile ----------------


@func
def _range_loop(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in range(8):
        y = relu(x)  # noqa: F841


def test_range_lowers_to_grid_region_scalar_iv():
    grid = _range_loop.body
    assert isinstance(grid, GridRegionExpr)
    # range(N) ≡ tile(N): start=0, extent=N, step=1 — the loop var is a scalar.
    assert grid.start == 0
    assert grid.extent == 8
    assert grid.step == 1
    assert isinstance(grid.induction_var, Var)


@func
def _range_start_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in range(2, 8, 3):
        y = relu(x)  # noqa: F841


def test_range_start_stop_step():
    grid = _range_start_step.body
    assert grid.start == 2
    assert grid.extent == 8   # `extent` is the stop endpoint (half-open [start, extent))
    assert grid.step == 3


# ── dim-expression loop bounds (e.g. C // N) --------------------------------


@func
def _dim_expr_extent(x: Tensor[(_SEQ, 4), "f32"]) -> Tensor[(_SEQ, 4), "f32"]:
    for i in tile(_SEQ // 2):
        y = relu(x)  # noqa: F841


def test_tile_accepts_dim_expression_extent():
    grid = _dim_expr_extent.body
    assert isinstance(grid, GridRegionExpr)
    # `_SEQ // 2` builds a DimFloorDiv dim Expr (a Call), not a bare DimVar.
    assert isinstance(grid.extent, Call)


# ── Nested GridRegions ------------------------------------------------------


@func
def _nested(x: Tensor[(8, 4), "f32"]) -> Tensor[(8, 4), "f32"]:
    o = relu(x)
    for r in range(8):
        for c in tile(4):
            o = add(o, x)
    return o


def test_nested_for_builds_nested_grid_region():
    outer = _nested.body
    assert isinstance(outer, GridRegionExpr)
    # `o` is bound before the outer loop and rebound only inside the inner
    # loop — the recursive carry scan still lifts it as the outer carry.
    assert len(outer.carried_args) == 1
    assert outer.carried_args[0].name == "o"
    # The outer loop's yield is the inner GridRegionExpr (the nested loop).
    assert isinstance(outer.yield_values[0], GridRegionExpr)
    inner = outer.yield_values[0]
    assert len(inner.carried_args) == 1
    assert inner.carried_args[0].name == "o"
