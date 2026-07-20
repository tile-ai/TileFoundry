"""End-to-end (parse + evaluate) tests for the unified ``range`` / ``tile``
loop surface, nested GridRegions, and dim-expression loop bounds.

``range`` and ``tile`` share one loop domain ``(start, extent, step)`` and lower
to the same ``GridRegionExpr``; they differ only in the loop-var binding
(``range`` → scalar, two-arg ``tile`` → slice). Neither is unrolled. Nested
``for`` loops produce nested GridRegions, and loop bounds accept dim expressions
(e.g. ``C // N``) resolved at evaluate time.
"""
from __future__ import annotations

import torch

from tilefoundry import func
from tilefoundry.dsl import Tensor, tf  # noqa: F401
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.types.dim import DimVar

_M = DimVar("m", 1, 64)
_K = DimVar("k", 1, 64)
_SUM = ReduceKind.SUM


@func
def _range_sum(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
    acc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
    acc = tf.full_like(acc, value=0.0)
    for i in range(_M):  # noqa: F821 — range over a DimVar extent
        acc = acc + tf.gather(x, i, axis=0)
    return acc


@func
def _range_start_step(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
    acc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
    acc = tf.full_like(acc, value=0.0)
    for i in range(1, _M, 2):  # noqa: F821 — odd indices
        acc = acc + tf.gather(x, i, axis=0)
    return acc


@func
def _nested_sum(x: Tensor[(_M, _K), "f32"]) -> Tensor[(), "f32"]:
    # `total` is bound before the outer loop and rebound ONLY inside the inner
    # loop — exercises the recursive carry scan + nested GridRegions.
    total = tf.reduce(x, axes=(0, 1), keepdim=False, kind=_SUM)
    total = tf.full_like(total, value=0.0)
    for r in range(_M):  # noqa: F821
        row = tf.gather(x, r, axis=0)
        for c in tile(_K):  # noqa: F821
            total = total + tf.gather(row, c, axis=0)
    return total


@func
def _dim_expr_half_sum(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
    acc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
    acc = tf.full_like(acc, value=0.0)
    for i in tile(_M // 2):  # noqa: F821 — dim-expression extent
        acc = acc + tf.gather(x, i, axis=0)
    return acc


def test_range_scalar_iv_sum():
    n = 5
    x = torch.arange(n, dtype=torch.float32)
    out = evaluate(_range_sum, x, device="cpu")
    assert torch.allclose(out.reshape(()), x.sum()), (n, out)


def test_range_start_step():
    n = 7
    x = torch.arange(n, dtype=torch.float32)
    out = evaluate(_range_start_step, x, device="cpu")
    assert torch.allclose(out.reshape(()), x[1:n:2].sum()), (n, out)


def test_nested_grid_region_outer_carry_in_inner():
    x = torch.randn(4, 5)
    out = evaluate(_nested_sum, x, device="cpu")
    assert torch.allclose(out.reshape(()), x.sum(), atol=1e-4), out


def test_dim_expression_extent():
    n = 8
    x = torch.arange(n, dtype=torch.float32)
    out = evaluate(_dim_expr_half_sum, x, device="cpu")
    assert torch.allclose(out.reshape(()), x[: n // 2].sum()), (n, out)


# ── interleaved two-partial reduction == flat reduction --------------------
# The split-KV decomposition relies on: partition the reduction axis across
# `NUM_SPLITS` partials (partial p owns indices ≡ p mod N), reduce each
# independently, then combine. For a plain sum this must equal the flat sum.

_NSPLIT = 2


@func
def _interleaved_two_partial_sum(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
    g = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
    g = tf.full_like(g, value=0.0)
    for p in range(_NSPLIT):  # noqa: F821 — outer: one partial per split
        pacc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
        pacc = tf.full_like(pacc, value=0.0)
        for i in tile(_M // _NSPLIT):  # noqa: F821 — inner: this partial's indices
            idx = p + i * _NSPLIT
            pacc = pacc + tf.gather(x, idx, axis=0)
        g = g + pacc  # combine partials
    return g


def test_interleaved_partial_reduction_equals_flat():
    n = 6  # multiple of NUM_SPLITS
    x = torch.randn(n)
    out = evaluate(_interleaved_two_partial_sum, x, device="cpu")
    assert torch.allclose(out.reshape(()), x.sum(), atol=1e-4), (n, out)
