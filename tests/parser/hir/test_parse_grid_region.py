"""``for`` over ``tile`` / ``range`` parses to a GridRegionExpr.

``range`` and two-arg ``tile`` share one loop domain ``(start, extent, step)``;
body Assigns whose LHS is an outer-scope Var get lifted to ``carried_args`` +
``yield_values``. The corpus authors a carried-accumulator ``tile`` loop and
evaluates it, so this file keeps the domain forms no model spells out and the
diagnostics for loop bodies the surface does not support.
"""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tilefoundry import func
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.core import Call, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.hir.grid_region import GridRegionExpr

_SEQ = DimVar("seq_len", 1, 100)

_PRELUDE = """from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor
"""


def _src(*body: str) -> str:
    """A one-``@func`` script over ``x``; *body* lines carry their own nesting."""
    lines = "\n".join(f"    {line}" for line in body)
    return f'{_PRELUDE}\n@func\ndef f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:\n{lines}\n'


@func
def _tile_default_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(8):
        y = relu(x)  # noqa: F841


@func
def _tile_extent_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(8, 2):
        y = relu(x)  # noqa: F841


@func
def _tile_dimvar_extent(x: Tensor[(_SEQ, 4), "f32"]) -> Tensor[(_SEQ, 4), "f32"]:
    for i in tile(_SEQ, 2):
        y = relu(x)  # noqa: F841


@func
def _tile_dim_expr_extent(x: Tensor[(_SEQ, 4), "f32"]) -> Tensor[(_SEQ, 4), "f32"]:
    for i in tile(_SEQ // 2):
        y = relu(x)  # noqa: F841


@func
def _range_start_stop_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in range(2, 8, 3):
        y = relu(x)  # noqa: F841


def test_iteration_domain_forms():
    """One loop domain behind both spellings: a single-arg ``tile`` steps by 1
    from 0, the second arg is the step, and the extent may be a static int, a
    ``DimVar``, or a dim expression (a ``Call``, not a bare DimVar). ``range``
    carries the start and binds a scalar induction var; its ``extent`` is the stop
    endpoint of the half-open ``[start, extent)`` domain. A loop that rebinds
    nothing outer carries nothing."""
    grid = _tile_default_step.body
    assert isinstance(grid, GridRegionExpr)
    assert (grid.start, grid.extent, grid.step) == (0, 8, 1)
    assert grid.carried_args == ()
    assert grid.init_args == ()
    assert grid.yield_values == ()

    assert (_tile_extent_step.body.extent, _tile_extent_step.body.step) == (8, 2)
    assert (_tile_dimvar_extent.body.extent, _tile_dimvar_extent.body.step) == (_SEQ, 2)
    assert isinstance(_tile_dim_expr_extent.body.extent, Call)

    ranged = _range_start_stop_step.body
    assert isinstance(ranged, GridRegionExpr)
    assert (ranged.start, ranged.extent, ranged.step) == (2, 8, 3)
    assert isinstance(ranged.induction_var, Var)


@func
def _single_carry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    o = relu(x)
    for i in tile(8):
        o = add(o, x)
    return o


@func
def _inner_only(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(8):
        t = relu(x)
        z = add(t, x)  # noqa: F841


def test_carry_lifting_is_scoped_to_outer_bindings():
    """Rebinding a Var bound before the loop lifts it to a phi carry whose yield
    is the rebinding RHS, and the loop's own type is the phi's. Names first bound
    inside the body are not outer-scope when scanned, so they create no carry slot
    — otherwise every temporary would become a loop-carried value."""
    grid = _single_carry.body
    assert isinstance(grid, GridRegionExpr)
    (phi,) = grid.carried_args
    assert isinstance(phi, Var)
    assert phi.name == "o"
    (yielded,) = grid.yield_values
    assert isinstance(yielded, Call)
    assert grid.type == phi.type

    assert _inner_only.body.carried_args == ()


@func
def _nested(x: Tensor[(8, 4), "f32"]) -> Tensor[(8, 4), "f32"]:
    o = relu(x)
    for r in range(8):
        for c in tile(4):
            o = add(o, x)
    return o


def test_nested_for_builds_nested_grid_region():
    """`o` is bound before the outer loop and rebound only inside the inner loop:
    the recursive carry scan still lifts it as the outer carry, and the outer
    loop's yield is the inner GridRegionExpr."""
    outer = _nested.body
    assert isinstance(outer, GridRegionExpr)
    assert [v.name for v in outer.carried_args] == ["o"]
    inner = outer.yield_values[0]
    assert isinstance(inner, GridRegionExpr)
    assert [v.name for v in inner.carried_args] == ["o"]


def test_tile_rejects_non_dim_expr():
    # A bare tensor (not int / DimVar / dim-op Expr) is not a legal extent.
    with pytest.raises(VerifyError, match="dim expression"):
        import_dsl(_src("for i in tile(x):", "    y = relu(x)"))


def test_return_inside_tile_body_rejected():
    with pytest.raises(VerifyError, match="must not contain `return`"):
        import_dsl(_src("for i in tile(8):", "    return x", "return x"))


def test_augassign_in_body_rejected():
    # v1 supports only `=`; an augmented assignment would hide a carry.
    with pytest.raises(VerifyError, match="augmented assignment"):
        import_dsl(_src("o = relu(x)", "for i in tile(8):", "    o += x", "return o"))
