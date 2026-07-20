"""GridRegionExpr dynamic (`ShapeDim`) extent / step — evaluator resolution.

`extent` / `step` may be a static `int` or a `ShapeDim` (a `DimVar` or a dim
`Expr`). A symbolic value is resolved to a concrete `int` at evaluate time from
the call's argument-shape DimVar bindings; the static-`int` path is unchanged.
Resolution fails closed on an unbound DimVar, a negative extent, or a
non-positive step.
"""
from __future__ import annotations

import pytest
import torch

from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.value import EvalError
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimSub, DimVar, ceildiv, simplify_dim


def _f32(shape):
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")


def _scalar_i32():
    return TensorType(shape=(), dtype=DType.i32, layout=None, storage="rmem")


def _sum_loop_fn(extent):
    """`acc = 0; for i in range(0, extent): acc += x[i]` over a `(seq_len,)` x."""
    N = DimVar("seq_len", 1, 100)
    x = Var(type=_f32((N,)), name="x")
    acc = Var(type=_f32(()), name="acc")
    iv = Var(type=_scalar_i32(), name="i")
    init = Constant(value=0.0, type=_f32(()))
    row = Call(type=_f32(()), target=Gather(axis=0), args=(x, iv))
    new_acc = Call(type=_f32(()), target=Binary(kind=BinaryKind.ADD), args=(acc, row))
    grid = GridRegionExpr(
        type=_f32(()), induction_var=iv, carried_args=(acc,),
        init_args=(init,), body=new_acc, yield_values=(new_acc,),
        extent=extent, step=1,
    )
    return Function.build(name="sumloop", params=(x,), body=grid, return_type=_f32(()))


# ── resolve_dim unit ──────────────────────────────────────────────────────


def test_resolve_dim_ceildiv():
    N = DimVar("seq_len", 1, 100)
    expr = ceildiv(N, 4)
    assert resolve_dim(expr, {"seq_len": 10}) == 3
    assert resolve_dim(expr, {"seq_len": 8}) == 2
    assert resolve_dim(N, {"seq_len": 7}) == 7
    assert resolve_dim(5, {}) == 5


# ── evaluator: DimVar extent resolves from the arg shape ──────────────────


def test_dimvar_extent_evaluates():
    fn = _sum_loop_fn(DimVar("seq_len", 1, 100))
    x = torch.randn(5)
    out = evaluate(fn, x, device="cpu")
    assert torch.allclose(out, x.sum())


def test_static_int_extent_unaffected():
    # Static extent path: loop runs exactly `extent` rows regardless of N.
    fn = _sum_loop_fn(4)
    x = torch.randn(10)
    out = evaluate(fn, x, device="cpu")
    assert torch.allclose(out, x[:4].sum())


def test_dynamic_step_evaluates():
    """`step` is a DimVar resolved from a parameter shape: the loop strides over
    `range(0, n, blk)` and sums `x[::blk]`."""
    n, blk = 8, 2
    N = DimVar("n", 1, 100)
    B = DimVar("blk", 1, 16)
    x = Var(type=_f32((N,)), name="x")
    stride_hint = Var(type=_f32((B,)), name="stride_hint")  # binds `blk` via its length
    acc = Var(type=_f32(()), name="acc")
    iv = Var(type=_scalar_i32(), name="i")
    init = Constant(value=0.0, type=_f32(()))
    row = Call(type=_f32(()), target=Gather(axis=0), args=(x, iv))
    new_acc = Call(type=_f32(()), target=Binary(kind=BinaryKind.ADD), args=(acc, row))
    grid = GridRegionExpr(
        type=_f32(()), induction_var=iv, carried_args=(acc,),
        init_args=(init,), body=new_acc, yield_values=(new_acc,),
        extent=N, step=B,
    )
    fn = Function.build(
        name="stridesum", params=(x, stride_hint), body=grid, return_type=_f32(()),
    )
    xv = torch.randn(n)
    out = evaluate(fn, xv, torch.zeros(blk), device="cpu")
    assert torch.allclose(out, xv[::blk].sum())


# ── fail closed ───────────────────────────────────────────────────────────


def test_unbound_dimvar_extent_fails_closed():
    fn = _sum_loop_fn(DimVar("not_a_param_dim", 1, 100))
    with pytest.raises(EvalError, match="unbound DimVar"):
        evaluate(fn, torch.randn(5), device="cpu")


def test_negative_extent_fails_closed():
    # extent = seq_len - 100 resolves negative for small seq_len.
    N = DimVar("seq_len", 1, 100)
    neg = simplify_dim(DimSub, (N, 100))
    fn = _sum_loop_fn(neg)
    with pytest.raises(EvalError, match="non-negative"):
        evaluate(fn, torch.randn(5), device="cpu")


def test_nonpositive_step_fails_closed():
    N = DimVar("seq_len", 1, 100)
    x = Var(type=_f32((N,)), name="x")
    acc = Var(type=_f32(()), name="acc")
    iv = Var(type=_scalar_i32(), name="i")
    init = Constant(value=0.0, type=_f32(()))
    row = Call(type=_f32(()), target=Gather(axis=0), args=(x, iv))
    new_acc = Call(type=_f32(()), target=Binary(kind=BinaryKind.ADD), args=(acc, row))
    grid = GridRegionExpr(
        type=_f32(()), induction_var=iv, carried_args=(acc,),
        init_args=(init,), body=new_acc, yield_values=(new_acc,),
        extent=N, step=0,
    )
    fn = Function.build(name="badstep", params=(x,), body=grid, return_type=_f32(()))
    with pytest.raises(EvalError, match="step must be positive"):
        evaluate(fn, torch.randn(5), device="cpu")
