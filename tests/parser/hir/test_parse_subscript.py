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
from tilefoundry.ir.core import Call, VerifyError
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.types.shard.shard_layout import ShardLayout

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
    with pytest.raises(VerifyError, match="tile.. takes 2 arguments"):
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


def test_runtime_start_slice_has_static_size_and_no_plain_layout_claim():
    src = _PRELUDE + (
        '\nfrom tilefoundry.ir.types.shard import Layout\n'
        'plain_layout = Layout((8, 4), (4, 1))\n'
        '\n@func\ndef f(x: Tensor[(8, 4), "f32", plain_layout], '
        'start: Tensor[(), "i64"]) -> Tensor[(4, 4), "f32"]:\n'
        "    return x[start:start + 4, :]\n"
    )
    fn = import_dsl(src)

    assert isinstance(fn.body, Call)
    assert isinstance(fn.body.target, Slice)
    assert fn.body.target.sizes == (4, 4)
    assert fn.body.args[1].elements[0] is fn.params[1]
    assert fn.body.type.layout is None

    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    start = torch.tensor(2, dtype=torch.int64)
    torch.testing.assert_close(
        evaluate(fn, x, start, device="cpu"), x[2:6, :]
    )


def test_runtime_start_slice_keeps_sharding_without_an_offset():
    src = _PRELUDE + (
        '\nfrom tilefoundry.ir.types.shard import Layout, Mesh, ShardLayout, Split, Topology\n'
        '\n@func\ndef f(x: Tensor[(8, 4), "f32", ShardLayout('
        'layout=Layout((8, 2, 2), (4, 2, 1)), attrs=(Split(1),), '
        'mesh=Mesh((Topology("gpu", 2),), Layout((2,), (1,)), names=("g",)))], '
        'start: Tensor[(), "i64"]) -> Tensor[(2, 4), "f32"]:\n'
        "    return x[start:start + 2, :]\n"
    )
    fn = import_dsl(src)

    assert isinstance(fn.body, Call)
    assert isinstance(fn.body.target, Slice)
    assert fn.body.target.sizes == (2, 4)
    assert isinstance(fn.body.type.layout, ShardLayout)
    assert fn.body.type.layout.attrs == fn.body.args[0].type.layout.attrs
    assert fn.body.type.layout.layout.shape == (2, 2, 2)


def test_runtime_slice_stride_is_still_compile_time():
    bad = _src(
        'x: Tensor[(8, 4), "f32"], step: Tensor[(), "i64"]'
        ') -> Tensor[(4, 4), "f32"]',
        "return x[0:8:step, :]",
    )
    with pytest.raises(
        VerifyError,
        match=r"tensor subscript axis 0: slice stride must be a compile-time dimension",
    ):
        import_dsl(bad)


def test_range_scalar_can_drive_a_manual_slice_window():
    fn = import_dsl(_src(
        'x: Tensor[(8, 4), "f32"]) -> Tensor[(2, 4), "f32"]',
        "out = x[0:2, :]",
        "for i in range(0, 8, 2):",
        "    out = x[i:i + 2, :]",
        "return out",
    ))

    x = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    torch.testing.assert_close(
        evaluate(fn, x, device="cpu"), x[6:8, :]
    )


def test_runtime_start_slice_rejects_an_unrelated_stop():
    bad = _src(
        'x: Tensor[(8, 4), "f32"], start: Tensor[(), "i64"]'
        ') -> Tensor[(4, 4), "f32"]',
        "return x[start:8, :]",
    )
    with pytest.raises(
        VerifyError,
        match=r"tensor subscript axis 0: a run-time start needs the stop endpoint",
    ):
        import_dsl(bad)


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


_HALF, _STEP, _ROWS = 4, 2, 3


@func
def _fused_halves(gu: Tensor[(_ROWS, 2 * _HALF), "f32"], seed: Tensor[(_ROWS, _STEP), "f32"]):
    out = add(seed, seed)
    for n in tile(_HALF, _STEP):
        out = add(out, mul(gu[:, n], gu[:, n + _HALF]))
    return out


@func
def _summed_offsets(gu: Tensor[(_ROWS, 2 * _HALF), "f32"], seed: Tensor[(_ROWS, _STEP), "f32"]):
    out = add(seed, seed)
    for n in tile(_HALF, _STEP):
        out = add(out, mul(gu[:, n], gu[:, _HALF + 1 + n - 1]))
    return out


def _fused_reference(gu, seed):
    out = seed * 2
    for lo in range(0, _HALF, _STEP):
        out = out + gu[:, lo:lo + _STEP] * gu[:, lo + _HALF:lo + _HALF + _STEP]
    return out


def test_a_compile_time_offset_moves_a_tile_window_without_resizing_it():
    """Two windows a fixed distance apart, read in one loop over one tensor.

    The fused ``[gate | up]`` read: the offset moves the base and leaves the
    length, so both reads have the same static shape and land ``_HALF`` columns
    apart. Offsets accumulate, so a sum of terms names the same move.
    """
    gu = torch.arange(_ROWS * 2 * _HALF, dtype=torch.float32).reshape(_ROWS, 2 * _HALF)
    seed = torch.ones((_ROWS, _STEP), dtype=torch.float32)
    expected = _fused_reference(gu, seed)

    assert _fused_halves.return_type.shape == (_ROWS, _STEP)
    torch.testing.assert_close(evaluate(_fused_halves, gu, seed, device="cpu"), expected)
    torch.testing.assert_close(evaluate(_summed_offsets, gu, seed, device="cpu"), expected)


def test_a_run_time_endpoint_and_a_moved_window_share_one_subscript():
    """One axis takes its start at run time while another moves its window.

    Both keep ``Slice.sizes`` static, by different routes: a run-time endpoint
    pairs with a compile-time window, and a move keeps the window it was given.
    """
    mixed = import_dsl(_src(
        'x: Tensor[(8, 8), "f32"], e: Tensor[(), "i64"], '
        f'seed: Tensor[(2, {_STEP}), "f32"]) -> Tensor[(2, {_STEP}), "f32"]',
        "out = add(seed, seed)",
        f"for n in tile({_HALF}, {_STEP}):",
        f"    out = add(out, x[e:e + 2, n + {_HALF}])",
        "return out",
    ))

    x = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    seed = torch.ones((2, _STEP), dtype=torch.float32)
    expected = seed * 2
    for lo in range(0, _HALF, _STEP):
        expected = expected + x[3:5, lo + _HALF:lo + _HALF + _STEP]

    torch.testing.assert_close(
        evaluate(mixed, x, torch.tensor(3, dtype=torch.int64), seed, device="cpu"),
        expected,
    )


def test_a_moved_window_round_trips_as_the_move_it_was_written_as():
    script = as_script(_fused_halves)
    assert f"gu[:, n + {_HALF}]" in script, script

    gu = torch.arange(_ROWS * 2 * _HALF, dtype=torch.float32).reshape(_ROWS, 2 * _HALF)
    seed = torch.ones((_ROWS, _STEP), dtype=torch.float32)
    torch.testing.assert_close(
        evaluate(import_dsl(script), gu, seed, device="cpu"),
        _fused_reference(gu, seed),
    )


def test_a_move_a_window_cannot_make_is_rejected():
    """Each way a move stops describing a window of the same length on the axis.

    Each way a move stops describing a window of the same length inside the
    axis: an offset that is not compile-time, a subtraction that reverses the
    window instead of moving it, a move that walks the last window off the end,
    and a move that starts the first window before the front. The extent, the
    window length and the offset are all compile-time, so the last two are
    answered here rather than at evaluate time.
    """
    runtime_offset = _src(
        'x: Tensor[(1, 8), "f32"], k: Tensor[(), "i64"]) -> Tensor[(1, 2), "f32"]',
        "o = relu(x[:, 0:2])",
        "for n in tile(4, 2):",
        "    o = relu(x[:, n + k])",
        "return o",
    )
    with pytest.raises(VerifyError, match="moves by a compile-time integer"):
        import_dsl(runtime_offset)

    reversed_window = _src(
        'x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 2), "f32"]',
        "o = relu(x[:, 0:2])",
        "for n in tile(4, 2):",
        "    o = relu(x[:, 4 - n])",
        "return o",
    )
    with pytest.raises(VerifyError, match="reverses the window"):
        import_dsl(reversed_window)

    off_the_end = _src(
        'x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 2), "f32"]',
        "o = relu(x[:, 0:2])",
        "for n in tile(4, 2):",
        "    o = relu(x[:, n + 5])",
        "return o",
    )
    with pytest.raises(VerifyError, match=r"reads \[7, 9\).*axis is 8 long"):
        import_dsl(off_the_end)

    before_the_front = _src(
        'x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 2), "f32"]',
        "o = relu(x[:, 0:2])",
        "for n in tile(4, 2):",
        "    o = relu(x[:, n - 2])",
        "return o",
    )
    with pytest.raises(VerifyError, match="begin before the axis"):
        import_dsl(before_the_front)


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

    scaled_window = _src(
        'x: Tensor[(1, 8), "f32"]) -> Tensor[(1, 2), "f32"]',
        "o = relu(x[:, 0:2])",
        "for n in tile(4, 2):",
        "    o = relu(x[:, n * 2])",
        "return o",
    )
    with pytest.raises(VerifyError, match="unsupported indexer"):
        import_dsl(scaled_window)
