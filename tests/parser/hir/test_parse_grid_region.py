"""``for`` over ``tile`` / ``range`` parses to a GridRegionExpr.

``range`` and ``tile`` share one loop domain ``(start, extent, step)``;
body Assigns whose LHS is an outer-scope Var get lifted to ``carried_args`` +
``yield_values``. The corpus authors a carried-accumulator grid loop and
evaluates it, so this file keeps the domain forms no model spells out and the
diagnostics for loop bodies the surface does not support.
"""

from __future__ import annotations

import ast
from dataclasses import replace

import pytest

from tests._source import import_dsl
from tilefoundry import func
from tilefoundry.analysis.walk import postorder
from tilefoundry.dsl import DimVar, Tensor
from tilefoundry.dsl._stub_gen import regen_stubs
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
def _range_default_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in range(8):
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
def _range_dim_expr_extent(x: Tensor[(_SEQ, 4), "f32"]) -> Tensor[(_SEQ, 4), "f32"]:
    for i in range(_SEQ // 2):
        y = relu(x)  # noqa: F841


@func
def _range_start_stop_step(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in range(2, 8, 3):
        y = relu(x)  # noqa: F841


def test_iteration_domain_forms():
    """One loop domain behind both spellings.

    ``range`` steps by 1 from 0 unless given a start or step. ``tile`` takes an
    explicit window step. Either extent may be a static int, a ``DimVar``, or a
    dim expression (a ``Call``, not a bare DimVar). ``range`` binds a scalar
    induction var; its ``extent`` is the stop endpoint of the half-open
    ``[start, extent)`` domain. A loop that rebinds nothing outer carries nothing.
    """
    grid = _range_default_step.body
    assert isinstance(grid, GridRegionExpr)
    assert (grid.start, grid.extent, grid.step) == (0, 8, 1)
    assert grid.carried_args == ()
    assert grid.init_args == ()
    assert grid.yield_values == ()
    assert repr(grid) == repr(replace(_tile_extent_step.body, step=1))

    assert (_tile_extent_step.body.extent, _tile_extent_step.body.step) == (8, 2)
    assert (_tile_dimvar_extent.body.extent, _tile_dimvar_extent.body.step) == (_SEQ, 2)
    assert isinstance(_range_dim_expr_extent.body.extent, Call)

    ranged = _range_start_stop_step.body
    assert isinstance(ranged, GridRegionExpr)
    assert (ranged.start, ranged.extent, ranged.step) == (2, 8, 3)
    assert isinstance(ranged.induction_var, Var)


@func
def _single_carry(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    o = relu(x)
    for i in range(8):
        o = add(o, x)
    return o


@func
def _inner_only(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in range(8):
        t = relu(x)
        z = add(t, x)  # noqa: F841


def test_carry_lifting_is_scoped_to_outer_bindings():
    """Test carry lifting is scoped to outer bindings.

    Rebinding a Var bound before the loop lifts it to a phi carry whose yield
    is the rebinding RHS, and the loop's own type is the phi's. Names first bound
    inside the body are not outer-scope when scanned, so they create no carry slot
    — otherwise every temporary would become a loop-carried value.
    """
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
def _carry_reads_old_and_new_values(x: Tensor[(8,), "f32"]):
    m = relu(x)
    o = relu(x)
    for i in range(8):
        m_new = maximum(m, x)
        correction = sub(m, m_new)
        o = add(o, correction)
        m = m_new
    return o


def test_carry_rebinding_reuses_the_existing_rhs_node() -> None:
    grid = _carry_reads_old_and_new_values.body.args[0]
    assert isinstance(grid, GridRegionExpr)
    carried = dict(zip((value.name for value in grid.carried_args), grid.carried_args))
    yielded = dict(zip((value.name for value in grid.carried_args), grid.yield_values))
    correction = yielded["o"].args[1]

    assert grid.body is yielded["m"]
    assert correction.args[0] is carried["m"]
    assert correction.args[1] is yielded["m"]


@func
def _carry_initialized_from_a_parameter(
    x: Tensor[(8,), "f32"],
) -> Tensor[(8,), "f32"]:
    acc = x
    for i in range(8):
        acc = add(acc, x)
    return acc


def test_a_bare_name_initializes_a_carry_without_adding_a_call() -> None:
    grid = _carry_initialized_from_a_parameter.body
    assert isinstance(grid, GridRegionExpr)
    assert grid.init_args[0] is _carry_initialized_from_a_parameter.params[0]
    assert [expr for expr in postorder(grid) if isinstance(expr, Call)] == [grid.body]


@func
def _nested(x: Tensor[(8, 4), "f32"]) -> Tensor[(8, 4), "f32"]:
    o = relu(x)
    for r in range(8):
        for c in range(4):
            o = add(o, x)
    return o


def test_nested_for_builds_nested_grid_region():
    """`o` is bound before the outer loop and rebound only inside the inner loop.

    `o` is bound before the outer loop and rebound only inside the inner loop:
    the recursive carry scan still lifts it as the outer carry, and the outer
    loop's yield is the inner GridRegionExpr.
    """
    outer = _nested.body
    assert isinstance(outer, GridRegionExpr)
    assert [v.name for v in outer.carried_args] == ["o"]
    inner = outer.yield_values[0]
    assert isinstance(inner, GridRegionExpr)
    assert [v.name for v in inner.carried_args] == ["o"]


def test_single_argument_tile_points_to_range():
    with pytest.raises(VerifyError, match=r"use range\(extent\)"):
        import_dsl(_src("for i in tile(8):", "    y = relu(x)"))


@pytest.mark.parametrize("loop", ["tile(8, step=2)", "range(stop=8)"])
def test_grid_loops_reject_keyword_args(loop: str):
    with pytest.raises(VerifyError, match="positional-only at the IR level"):
        import_dsl(_src(f"for i in {loop}:", "    y = relu(x)"))


def test_generated_tile_stub_requires_window_step(tmp_path):
    stub = ast.parse(regen_stubs(tmp_path)["tf"].read_text())
    tile_def = next(
        node for node in stub.body if isinstance(node, ast.FunctionDef) and node.name == "tile"
    )
    assert [arg.arg for arg in tile_def.args.args] == ["extent", "step"]
    assert tile_def.args.defaults == []


def test_range_rejects_non_dim_expr():

    with pytest.raises(VerifyError, match="dim expression"):
        import_dsl(_src("for i in range(x):", "    y = relu(x)"))


def test_return_inside_grid_body_rejected():
    with pytest.raises(VerifyError, match="must not contain `return`"):
        import_dsl(_src("for i in range(8):", "    return x", "return x"))


def test_augassign_in_body_rejected():

    with pytest.raises(VerifyError, match="augmented assignment"):
        import_dsl(_src("o = relu(x)", "for i in range(8):", "    o += x", "return o"))
