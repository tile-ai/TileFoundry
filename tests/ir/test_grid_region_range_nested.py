"""Define test grid region range nested behavior.

End-to-end (parse + evaluate) tests for the unified ``range`` / ``tile``
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
def _range_start_step(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
    acc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
    acc = tf.full_like(acc, value=0.0)
    for i in range(1, _M, 2):  # noqa: F821 — odd indices
        selected = tf.index_select(x, tf.reshape(i, new_shape=(1,)), dim=0)
        acc = acc + tf.reshape(selected, new_shape=())
    return acc


@func
def _nested_sum(x: Tensor[(_M, _K), "f32"]) -> Tensor[(), "f32"]:

    total = tf.reduce(x, axes=(0, 1), keepdim=False, kind=_SUM)
    total = tf.full_like(total, value=0.0)
    for r in range(_M):  # noqa: F821
        selected_row = tf.index_select(x, tf.reshape(r, new_shape=(1,)), dim=0)
        row = tf.reshape(selected_row, new_shape=(_K,))
        for c in tile(_K):  # noqa: F821
            selected = tf.index_select(row, tf.reshape(c, new_shape=(1,)), dim=0)
            total = total + tf.reshape(selected, new_shape=())
    return total


@func
def _dim_expr_half_sum(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
    acc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
    acc = tf.full_like(acc, value=0.0)
    for i in tile(_M // 2):  # noqa: F821 — dim-expression extent
        selected = tf.index_select(x, tf.reshape(i, new_shape=(1,)), dim=0)
        acc = acc + tf.reshape(selected, new_shape=())
    return acc


def test_range_start_step():
    """The three-argument `range` surface.

    The three-argument `range` surface: a scalar loop var over a DimVar extent
    with a non-unit start and step, none of which is unrolled.
    """
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
            selected = tf.index_select(x, tf.reshape(idx, new_shape=(1,)), dim=0)
            pacc = pacc + tf.reshape(selected, new_shape=())
        g = g + pacc
    return g


def test_interleaved_partial_reduction_equals_flat():
    n = 6
    x = torch.randn(n)
    out = evaluate(_interleaved_two_partial_sum, x, device="cpu")
    assert torch.allclose(out.reshape(()), x.sum(), atol=1e-4), (n, out)
