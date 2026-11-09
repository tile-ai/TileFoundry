"""LoopRegion dynamic (`ShapeDim`) extent / step — evaluator resolution.

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

from tilefoundry import services
from tilefoundry.interfaces import evaluator
from tilefoundry.ir import core, dim, expr, types
from tilefoundry.ir.hir import math, tensor


def _f32(shape):
    return types.TensorType(shape=shape, dtype=expr.DType.f32, layout=None, storage="gmem")


def _i32(shape, storage="rmem"):
    return types.TensorType(shape=shape, dtype=expr.DType.i32, layout=None, storage=storage)


def _sum_loop_fn(extent, *, step=1, extra_params=()):
    """`acc = 0; for i in range(0, extent, step).

    `acc = 0; for i in range(0, extent, step): acc += x[i]` over a
    `(seq_len,)` x, plus any *extra_params* whose lengths bind further DimVars.
    """
    N = expr.DimVar("seq_len", 1, 99)
    x = expr.Var(type=_f32((N,)), name="x")
    acc = expr.Var(type=_f32(()), name="acc")
    iv = expr.Var(type=_i32(()), name="i")
    init = expr.Constant(value=0.0, type=_f32(()))
    index = expr.Call(type=_i32((1,)), target=tensor.Reshape(new_shape=(1,)), args=(iv,))
    captured_x = expr.Var(type=x.type, name="x")
    selected = expr.Call(type=_f32((1,)), target=tensor.IndexSelect(dim=0), args=(captured_x, index))
    row = expr.Call(type=_f32(()), target=tensor.Reshape(new_shape=()), args=(selected,))
    new_acc = expr.Call(type=_f32(()), target=math.Binary(kind=expr.BinaryKind.ADD), args=(acc, row))
    grid = core.LoopRegion(
        type=_f32(()),
        induction_var=iv,
        params=(acc, captured_x),
        args=(init, x),
        body=new_acc,
        yield_values=(new_acc,),
        extent=extent,
        step=step,
    )
    return core.Function.from_parts(
        name="sumloop",
        params=(x, *extra_params),
        body=grid,
        return_type=_f32(()),
    )


def test_dynamic_bounds_resolve_from_shapes():
    """A dynamic extent and step resolve from the argument shapes.

    A DimVar or dim `Expr` in either position resolves against the bindings the
    call's argument shapes provide, and nothing else: `resolve_dim` is given the
    same environment the evaluator builds.
    """
    N = expr.DimVar("seq_len", 1, 99)
    assert evaluator.resolve_dim(dim.ceildiv(N, 4), {"seq_len": 10}) == 3
    assert evaluator.resolve_dim(dim.ceildiv(N, 4), {"seq_len": 8}) == 2
    assert evaluator.resolve_dim(N, {"seq_len": 7}) == 7
    assert evaluator.resolve_dim(5, {}) == 5

    x = torch.randn(5)
    assert torch.allclose(evaluator.evaluate(_sum_loop_fn(N), x), x.sum())

    blk = 2
    B = expr.DimVar("blk", 1, 15)
    stride_hint = expr.Var(type=_f32((B,)), name="stride_hint")
    fn = _sum_loop_fn(N, step=B, extra_params=(stride_hint,))
    xv = torch.randn(8)
    out = evaluator.evaluate(fn, xv, torch.zeros(blk))
    assert torch.allclose(out, xv[::blk].sum())


def test_dynamic_bounds_fail_closed():
    N = expr.DimVar("seq_len", 1, 99)

    with pytest.raises(evaluator.EvalError, match="unbound DimVar"):
        evaluator.evaluate(_sum_loop_fn(expr.DimVar("not_a_param_dim", 1, 99)), torch.randn(5))

    with pytest.raises(evaluator.EvalError, match="non-negative"):
        evaluator.evaluate(_sum_loop_fn(services.simplify_dim("sub", (N, 100))), torch.randn(5))

    with pytest.raises(evaluator.EvalError, match="step must be positive"):
        evaluator.evaluate(_sum_loop_fn(N, step=0), torch.randn(5))
