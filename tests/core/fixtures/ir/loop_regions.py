"""Loops over ``range`` with a start and step, nested, and bounded by dim expressions."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.target import CpuTarget

_M = DimVar("m", 1, 63)
_K = DimVar("k", 1, 63)
_SUM = ReduceKind.SUM
_NSPLIT = 2


@module(entry="range_start_step", target=CpuTarget())
class RangeStartStep:
    @func
    def range_start_step(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
        acc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
        acc = tf.full_like(acc, value=0.0)
        for i in range(1, _M, 2):  # noqa: F821 — odd indices
            selected = tf.index_select(x, tf.reshape(i, new_shape=(1,)), dim=0)
            acc = acc + tf.reshape(selected, new_shape=())
        return acc


@module(entry="nested_sum", target=CpuTarget())
class NestedSum:
    @func
    def nested_sum(x: Tensor[(_M, _K), "f32"]) -> Tensor[(), "f32"]:
        total = tf.reduce(x, axes=(0, 1), keepdim=False, kind=_SUM)
        total = tf.full_like(total, value=0.0)
        for r in range(_M):  # noqa: F821
            selected_row = tf.index_select(x, tf.reshape(r, new_shape=(1,)), dim=0)
            row = tf.reshape(selected_row, new_shape=(_K,))
            for c in range(_K):  # noqa: F821
                selected = tf.index_select(row, tf.reshape(c, new_shape=(1,)), dim=0)
                total = total + tf.reshape(selected, new_shape=())
        return total


@module(entry="dim_expr_half_sum", target=CpuTarget())
class DimExprHalfSum:
    @func
    def dim_expr_half_sum(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
        acc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
        acc = tf.full_like(acc, value=0.0)
        for i in range(_M // 2):  # noqa: F821 — dim-expression extent
            selected = tf.index_select(x, tf.reshape(i, new_shape=(1,)), dim=0)
            acc = acc + tf.reshape(selected, new_shape=())
        return acc


@module(entry="interleaved_two_partial_sum", target=CpuTarget())
class InterleavedTwoPartialSum:
    @func
    def interleaved_two_partial_sum(x: Tensor[(_M,), "f32"]) -> Tensor[(), "f32"]:
        g = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
        g = tf.full_like(g, value=0.0)
        for p in range(_NSPLIT):  # noqa: F821 — outer: one partial per split
            pacc = tf.reduce(x, axes=(0,), keepdim=False, kind=_SUM)
            pacc = tf.full_like(pacc, value=0.0)
            for i in range(_M // _NSPLIT):  # noqa: F821 — inner: this partial's indices
                idx = p + i * _NSPLIT
                selected = tf.index_select(x, tf.reshape(idx, new_shape=(1,)), dim=0)
                pacc = pacc + tf.reshape(selected, new_shape=())
            g = g + pacc
        return g
