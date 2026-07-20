"""Parser ``expr[idx]`` subscript dispatch + RangeSlice lift.

Two dispatch paths from the surface ``expr[...]`` syntax:

- TupleType (a call returning a tuple) → ``TupleGetItem(index=i)``.
- TensorType → a ``Slice`` Op call. ``tile(extent, step)`` returns a
  parser-side ``RangeSlice``; using the loop var inside a tensor
  subscript (``x[:, ok]``) lifts to a ``Slice`` with bounds
  ``[ok.start, ok.stop)``.

Plus the negative paths: unsupported indexers, non-constant tuple
index, tile arity, and subscript rank mismatch.
"""

from __future__ import annotations

import textwrap

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare bindings used by @func bodies
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.dim import DimAdd, DimMul
from tilefoundry.parser.hir_parser import parse_script


def _dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def _find_slice_call(grid: GridRegionExpr) -> Call:
    """Return the first Slice Call in *grid.body* (recursing into args)."""

    def walk(expr):
        if isinstance(expr, Call):
            if isinstance(expr.target, Slice):
                return expr
            for a in expr.args:
                hit = walk(a)
                if hit is not None:
                    return hit
        return None

    hit = walk(grid.body)
    assert hit is not None, "no Slice Call found in grid body"
    return hit


# ── Fixtures ──────────────────────────────────────────────────────────────


@func
def _chunked_subscript(x: Tensor[(1, 2048), "f32"]) -> Tensor[(1, 2048), "f32"]:
    o = relu(x)
    for ok in tile(2048, 512):
        o = relu(x[:, ok])
    return o


@func
def _partial_slice(x: Tensor[(1, 2048), "f32"]) -> Tensor[(1, 2048), "f32"]:
    o = relu(x)
    for ok in tile(2048, 512):
        o = relu(x[0:1, ok])
    return o


@func
def _quant_subscript(x: Tensor[(1, 1536), "bf16"]) -> Tensor[(1, 12), "f32"]:
    out = quant(x)
    x_scale = out[1]
    return x_scale


# ── Tests ─────────────────────────────────────────────────────────────────


def test_chunked_tile_lifts_subscript_to_slice():
    fn = _chunked_subscript
    grid = fn.body
    assert isinstance(grid, GridRegionExpr)
    sl = _find_slice_call(grid)
    sl_op = sl.target
    assert isinstance(sl_op, Slice)
    # axis 0 is `:` → begin=0, end=1 (full extent)
    assert isinstance(sl_op.begin[0], Constant) and sl_op.begin[0].value == 0
    assert isinstance(sl_op.end[0], Constant) and sl_op.end[0].value == 1
    # axis 1 is `ok` (RangeSlice) → begin = iv * step, end = iv*step + step
    begin1 = sl_op.begin[1]
    end1 = sl_op.end[1]
    assert isinstance(begin1, Call) and isinstance(begin1.target, DimMul)
    assert isinstance(end1, Call) and isinstance(end1.target, DimAdd)
    # All strides default to 1
    assert all(isinstance(s, Constant) and s.value == 1 for s in sl_op.strides)


def test_partial_slice_with_range_slice():
    """Mix of partial slice (`0:1`) on one axis and RangeSlice on another."""
    fn = _partial_slice
    grid = fn.body
    sl = _find_slice_call(grid)
    sl_op = sl.target
    # axis 0: 0:1
    assert isinstance(sl_op.begin[0], Constant) and sl_op.begin[0].value == 0
    assert isinstance(sl_op.end[0], Constant) and sl_op.end[0].value == 1
    # axis 1: RangeSlice — symbolic
    assert isinstance(sl_op.begin[1], Call)
    assert isinstance(sl_op.end[1], Call)


def test_tile_with_too_many_args_rejected():
    bad = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]:
    for i in tile(1, 2, 3):
        y = relu(x)
"""
    with pytest.raises(VerifyError, match="tile.. takes 1 or 2 arguments"):
        parse_script(_dedent(bad))


def test_subscript_rank_mismatch_rejected():
    bad = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(1, 2048), "f32"]) -> Tensor[(1, 2048), "f32"]:
    o = relu(x)
    for ok in tile(2048, 512):
        o = relu(x[ok])
    return o
"""
    with pytest.raises(VerifyError, match="rank 1 != tensor rank 2"):
        parse_script(_dedent(bad))


def test_tuple_subscript_emits_tuple_get_item() -> None:
    """``call_returning_tuple()[i]`` → ``TupleGetItem(index=i)`` with field dtype."""
    fn = _quant_subscript
    body = fn.body
    assert isinstance(body, Call) and isinstance(body.target, TupleGetItem)
    assert body.target.index == 1
    assert body.type.dtype == DType.f32


def test_subscript_errors_for_unsupported_indexers() -> None:
    """Tensor integer indexing + non-constant tuple index both raise."""
    bad_tensor = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(a: Tensor[(1, 4), "f32"], b: Tensor[(1, 4), "f32"]) -> Tensor[(1, 4), "f32"]:
    c = add(a, b)
    return c[0, 0]
"""
    with pytest.raises(VerifyError, match="unsupported indexer"):
        parse_script(_dedent(bad_tensor))

    bad_tuple = """
from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor

@func
def f(x: Tensor[(1, 1536), "bf16"], i: Tensor[(), "i64"]) -> Tensor[(1, 1536), "fp8e4m3"]:
    out = quant(x)
    return out[i]
"""
    with pytest.raises(VerifyError, match="integer constant index"):
        parse_script(_dedent(bad_tuple))
