"""Parser ``expr[idx]`` subscript dispatch + RangeSlice lift.

``expr[...]`` dispatches on the subject: a TupleType (a call returning a tuple)
yields ``TupleGetItem(index=i)``, a TensorType a ``Slice`` Op call, and a
compile-time list the element it holds. The tuple path is exercised wherever a
model unpacks a multi-output op; the RangeSlice lift — two-arg ``tile`` binds the
loop var to a range, and using it in a subscript slices the chunk that iteration
owns — is spelled out only here, together with integer indexing, which drops its
axis, and the indexers the surface refuses.
"""

from __future__ import annotations

import pytest

from tests._source import import_dsl
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare bindings used by @func bodies
from tilefoundry.ir.core import VerifyError
from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types.dim import DimAdd, DimMul

_PRELUDE = """from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor
"""


def _src(signature: str, *body: str) -> str:
    """A one-``@func`` script: *signature* closes the param list and states the
    return annotation, *body* lines carry their own nesting."""
    lines = "\n".join(f"    {line}" for line in body)
    return f"{_PRELUDE}\n@func\ndef f({signature}:\n{lines}\n"


def _slice_op(fn) -> Slice:
    """The ``Slice`` op of the first Slice Call in *fn*'s grid body."""
    grid = fn.body
    assert isinstance(grid, GridRegionExpr)
    stack = [grid.body]
    while stack:
        expr = stack.pop()
        if isinstance(expr, Call):
            if isinstance(expr.target, Slice):
                return expr.target
            stack.extend(expr.args)
    raise AssertionError("no Slice Call found in grid body")


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


def test_tile_loop_var_in_a_subscript_lifts_to_a_slice():
    """A ``:`` axis becomes the full static extent, an authored ``0:1`` keeps its
    constant bounds, and the loop var's axis becomes symbolic bounds computed from
    the induction var (``iv * step`` .. ``iv * step + step``) — the chunk the
    iteration owns, not a copy of the whole axis. Strides default to 1."""
    chunked = _slice_op(_chunked_subscript)
    assert isinstance(chunked.begin[0], Constant) and chunked.begin[0].value == 0
    assert isinstance(chunked.end[0], Constant) and chunked.end[0].value == 1
    assert isinstance(chunked.begin[1], Call) and isinstance(chunked.begin[1].target, DimMul)
    assert isinstance(chunked.end[1], Call) and isinstance(chunked.end[1].target, DimAdd)
    assert all(isinstance(s, Constant) and s.value == 1 for s in chunked.strides)

    partial = _slice_op(_partial_slice)
    assert isinstance(partial.begin[0], Constant) and partial.begin[0].value == 0
    assert isinstance(partial.end[0], Constant) and partial.end[0].value == 1
    assert isinstance(partial.begin[1], Call)
    assert isinstance(partial.end[1], Call)


def test_tile_with_too_many_args_rejected():
    bad = _src(
        'x: Tensor[(8,), "f32"]) -> Tensor[(8,), "f32"]',
        "for i in tile(1, 2, 3):",
        "    y = relu(x)",
    )
    with pytest.raises(VerifyError, match="tile.. takes 1 or 2 arguments"):
        import_dsl(bad)


@func
def _collapsed(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4), "f32"]:
    return x[:, :, 3]


@func
def _kept(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 1), "f32"]:
    return x[:, :, 3:4]


@func
def _counted_back(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4), "f32"]:
    return x[:, :, -1]


def test_an_integer_index_drops_its_axis_and_a_slice_keeps_it():
    """``x[..., 3]`` and ``x[..., 3:4]`` select the same element and differ only in
    rank, and ``-1`` counts from the extent — the distinctions torch draws. The
    dropped form is a one-element ``Slice`` reshaped."""
    assert _collapsed.body.type.shape == (1, 4)
    assert _kept.body.type.shape == (1, 4, 1)
    assert _counted_back.body.type.shape == (1, 4)

    sliced = _collapsed.body.args[0].target
    assert isinstance(sliced, Slice)
    assert isinstance(sliced.begin[2], Constant) and sliced.begin[2].value == 3
    assert isinstance(sliced.end[2], Constant) and sliced.end[2].value == 4
    assert _counted_back.body.args[0].target.begin[2].value == 7


def test_a_compile_time_list_is_indexed_where_it_is_written():
    """A comprehension and a plain list literal both bind a Python list of Exprs;
    indexing either picks an expression rather than emitting an op."""
    taps = _PRELUDE + (
        '\n@func\ndef f(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 1), "f32"]:\n'
        "    taps = [x[:, :, j:j + 1] for j in range(4)]\n"
        "    return add(taps[0], taps[-1])\n"
    )
    first, last = import_dsl(taps).body.args
    assert isinstance(first.target, Slice) and isinstance(last.target, Slice)
    assert first.target.begin[2].value == 0
    assert last.target.begin[2].value == 3

    literal = _PRELUDE + (
        '\n@func\ndef f(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 1), "f32"]:\n'
        "    ends = [x[:, :, 0:1], x[:, :, 7:8]]\n"
        "    return add(ends[0], ends[1])\n"
    )
    low, high = import_dsl(literal).body.args
    assert isinstance(low.target, Slice) and isinstance(high.target, Slice)
    assert low.target.begin[2].value == 0
    assert high.target.begin[2].value == 7


def test_unsupported_subscripts_are_rejected():
    """Two shapes of illegal indexer, each named by its own diagnostic: a subscript
    whose rank does not match the tensor's, and a runtime value as a tuple index
    (the field must be known at parse time to give the result a type)."""
    rank_mismatch = _src(
        'x: Tensor[(1, 2048), "f32"]) -> Tensor[(1, 2048), "f32"]',
        "o = relu(x)",
        "for ok in tile(2048, 512):",
        "    o = relu(x[ok])",
        "return o",
    )
    with pytest.raises(VerifyError, match="rank 1 != tensor rank 2"):
        import_dsl(rank_mismatch)

    runtime_tuple_index = _src(
        'x: Tensor[(1, 1536), "bf16"], i: Tensor[(), "i64"])'
        ' -> Tensor[(1, 1536), "fp8e4m3"]',
        "out = quant(x)",
        "return out[i]",
    )
    with pytest.raises(VerifyError, match="integer constant index"):
        import_dsl(runtime_tuple_index)
