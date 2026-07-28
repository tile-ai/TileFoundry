"""GridRegionExpr dynamic (`ShapeDim`) extent / step — evaluator resolution.

`extent` / `step` may be a static `int` or a `ShapeDim` (a `DimVar` or a dim
`Expr`). A symbolic value is resolved to a concrete `int` at evaluate time from
the call's argument-shape DimVar bindings. The failures are the reason this is a
unit test: an unbound DimVar, a negative extent or a non-positive step each have
a plausible-looking wrong answer (skip the loop, run it once, loop forever), so
resolution must refuse rather than choose one.
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


def _sum_loop_fn(extent, *, step=1, extra_params=()):
    """`acc = 0; for i in range(0, extent, step): acc += x[i]` over a
    `(seq_len,)` x, plus any *extra_params* whose lengths bind further DimVars."""
    N = DimVar("seq_len", 1, 100)
    x = Var(type=_f32((N,)), name="x")
    acc = Var(type=_f32(()), name="acc")
    iv = Var(type=TensorType(shape=(), dtype=DType.i32, layout=None, storage="rmem"), name="i")
    init = Constant(value=0.0, type=_f32(()))
    row = Call(type=_f32(()), target=Gather(axis=0), args=(x, iv))
    new_acc = Call(type=_f32(()), target=Binary(kind=BinaryKind.ADD), args=(acc, row))
    grid = GridRegionExpr(
        type=_f32(()), induction_var=iv, carried_args=(acc,),
        init_args=(init,), body=new_acc, yield_values=(new_acc,),
        extent=extent, step=step,
    )
    return Function.build(
        name="sumloop", params=(x, *extra_params), body=grid, return_type=_f32(()),
    )


def test_dynamic_extent_and_step_resolve_from_the_argument_shapes():
    """A DimVar or dim `Expr` in either position resolves against the bindings the
    call's argument shapes provide, and nothing else: `resolve_dim` is given the
    same environment the evaluator builds."""
    N = DimVar("seq_len", 1, 100)
    assert resolve_dim(ceildiv(N, 4), {"seq_len": 10}) == 3
    assert resolve_dim(ceildiv(N, 4), {"seq_len": 8}) == 2
    assert resolve_dim(N, {"seq_len": 7}) == 7
    assert resolve_dim(5, {}) == 5

    # extent = seq_len, read off the length of x.
    x = torch.randn(5)
    assert torch.allclose(evaluate(_sum_loop_fn(N), x, device="cpu"), x.sum())

    # step is a second DimVar, bound by the length of a second parameter, so the
    # loop strides over `range(0, n, blk)`.
    blk = 2
    B = DimVar("blk", 1, 16)
    stride_hint = Var(type=_f32((B,)), name="stride_hint")  # binds `blk` via its length
    fn = _sum_loop_fn(N, step=B, extra_params=(stride_hint,))
    xv = torch.randn(8)
    out = evaluate(fn, xv, torch.zeros(blk), device="cpu")
    assert torch.allclose(out, xv[::blk].sum())


def test_dynamic_bounds_fail_closed():
    N = DimVar("seq_len", 1, 100)

    # No parameter shape binds this name, so there is no extent to run.
    with pytest.raises(EvalError, match="unbound DimVar"):
        evaluate(_sum_loop_fn(DimVar("not_a_param_dim", 1, 100)), torch.randn(5), device="cpu")

    # extent = seq_len - 100 resolves negative for small seq_len.
    with pytest.raises(EvalError, match="non-negative"):
        evaluate(_sum_loop_fn(simplify_dim(DimSub, (N, 100))), torch.randn(5), device="cpu")

    with pytest.raises(EvalError, match="step must be positive"):
        evaluate(_sum_loop_fn(N, step=0), torch.randn(5), device="cpu")
