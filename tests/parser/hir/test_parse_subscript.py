"""Parser ``expr[idx]`` subscript dispatch + tile-window slice lift.

``expr[...]`` dispatches on the subject: tuples select a field, tensors select a
region, and compile-time lists select the expression they hold. Integer indexing
drops its axis while a slice keeps it. Tile windows are verified through their
runtime and analysis behavior rather than parser node shape.
"""

from __future__ import annotations

import pytest
import torch

from tests._source import import_dsl
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare bindings used by @func bodies
from tilefoundry.evaluator import EvalError, evaluate
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import VerifyError

_PRELUDE = """from tilefoundry import func
from tilefoundry.dsl.tf import *
from tilefoundry.dsl import Tensor
"""


def _src(signature: str, *body: str) -> str:
    """A one-``@func`` script.

    A one-``@func`` script: *signature* closes the param list and states the
    return annotation, *body* lines carry their own nesting.
    """
    lines = "\n".join(f"    {line}" for line in body)
    return f"{_PRELUDE}\n@func\ndef f({signature}:\n{lines}\n"


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


@func
def _strided_and_clamped(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 3), "f32"]:
    return x[:, :, 1:20:3]


def test_an_integer_index_drops_its_axis_and_a_slice_keeps_it():
    """``x[..., 3]`` and ``x[..., 3:4]`` select the same element and differ only in rank.

    ``x[..., 3]`` and ``x[..., 3:4]`` select the same element and differ only in
    rank, and ``-1`` counts from the extent — the distinctions torch draws. The
    dropped form is a one-element ``Slice`` reshaped.
    """
    x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    torch.testing.assert_close(evaluate(_collapsed, x, device="cpu"), x[:, :, 3])
    torch.testing.assert_close(evaluate(_kept, x, device="cpu"), x[:, :, 3:4])
    torch.testing.assert_close(evaluate(_counted_back, x, device="cpu"), x[:, :, -1])
    torch.testing.assert_close(
        evaluate(_strided_and_clamped, x, device="cpu"), x[:, :, 1:20:3]
    )


def test_a_compile_time_list_is_indexed_where_it_is_written():
    """A comprehension and a plain list literal both bind a Python list of Exprs.

    A comprehension and a plain list literal both bind a Python list of Exprs;
    indexing either picks an expression rather than emitting an op.
    """
    taps = _PRELUDE + (
        '\n@func\ndef f(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 1), "f32"]:\n'
        "    taps = [x[:, :, j:j + 1] for j in range(4)]\n"
        "    return add(taps[0], taps[-1])\n"
    )
    x = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    torch.testing.assert_close(
        evaluate(import_dsl(taps), x, device="cpu"),
        x[:, :, 0:1] + x[:, :, 3:4],
    )

    literal = _PRELUDE + (
        '\n@func\ndef f(x: Tensor[(1, 4, 8), "f32"]) -> Tensor[(1, 4, 1), "f32"]:\n'
        "    ends = [x[:, :, 0:1], x[:, :, 7:8]]\n"
        "    return add(ends[0], ends[1])\n"
    )
    torch.testing.assert_close(
        evaluate(import_dsl(literal), x, device="cpu"),
        x[:, :, 0:1] + x[:, :, 7:8],
    )


@func
def _tail_window(x: Tensor[(10, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
    out = add(seed, seed)
    for row in tile(10, 4):
        out = add(x[row, :], seed)
    return out


def test_non_divisible_tile_window_evaluation_fails_closed():
    with pytest.raises(EvalError, match="Slice window exceeds axis 0"):
        evaluate(
            _tail_window,
            torch.ones((10, 4)),
            torch.ones((4, 4)),
            device="cpu",
        )


@func
def _full_window_roundtrip(x: Tensor[(8, 4), "f32"], seed: Tensor[(4, 4), "f32"]):
    out = add(seed, seed)
    for row in tile(8, 4):
        out = add(x[row, :], seed)
    return out


def test_tile_window_canonical_roundtrip_preserves_evaluation():
    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    seed = torch.ones((4, 4), dtype=torch.float32)
    expected = evaluate(_full_window_roundtrip, x, seed, device="cpu")

    restored = import_dsl(as_script(_full_window_roundtrip))
    torch.testing.assert_close(evaluate(restored, x, seed, device="cpu"), expected)


def test_unsupported_subscripts_are_rejected():
    """Two shapes of illegal indexer, each named by its own diagnostic.

    Two shapes of illegal indexer, each named by its own diagnostic: a subscript
    whose rank does not match the tensor's, and a runtime value as a tuple index
    (the field must be known at parse time to give the result a type).
    """
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
        'x: Tensor[(1, 1536), "bf16"], i: Tensor[(), "i64"]) -> Tensor[(1, 1536), "fp8e4m3"]',
        "out = quant(x)",
        "return out[i]",
    )
    with pytest.raises(VerifyError, match="integer constant index"):
        import_dsl(runtime_tuple_index)
