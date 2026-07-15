"""RMSNorm tests: typeinfer relaxation + HIR->TIR lowering + CUDA codegen.

Typeinfer (``hir.RMSNorm``):
``rms_norm`` was previously restricted to rank-2 ``x`` with
``x.dtype == weight.dtype``. Qwen3 shapes (``[1, 1, 2048]`` ``bf16``
input + ``[2048]`` ``f32`` weight) require rank-N + dtype-mismatch
acceptance. This file locks the relaxed contract.

Lowering / codegen (Qwen3 RMSNorm — two expression styles, same SSA pipeline):

Builder A: declarative reduce op composition
  sq = mul(x, x) / mean = reduce(sq, axes=(-1,), kind=MEAN)
  / add(mean, eps) / rsqrt / mul(x, scale) / mul(result, weight) / cast

Builder B: GridRegion per-row using Slice → TensorView
  For each row: slice(x, m) → row (K,) / cast f32 / square
  / reduce_mean → scalar / add eps / rsqrt / mul scalar broadcast
  / mul weight / cast bf16 / Copy to output[m,:] via TensorView
"""

from __future__ import annotations

import pytest
import torch

from tests.ops.eval_utils import EvalCase, run_eval_case
from tests.ops.typeinfer_utils import (
    ExpectedError,
    TypeInferCase,
    mesh,
    run_typeinfer_case,
    sharded,
)
from tilefoundry.dsl import DimVar
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.hir.math.unary import Unary
from tilefoundry.ir.hir.nn.rms_norm import RMSNorm
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.gather import Gather
from tilefoundry.ir.hir.tensor.reduce import Reduce, ReduceKind
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmt import Stmt
from tilefoundry.ir.tir.stmts import (
    Evaluate,
    For,
    If,
    LetStmt,
    MeshScope,
    Sequential,
    While,
)
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.ir.types.shard.shard_layout import Partial
from tilefoundry.passes.transforms.hir_to_tir import HirToTirPass


def _ten(shape, dtype):
    return TensorType(shape=shape, dtype=dtype, layout=None, storage="gmem")


# ---------------------------------------------------------------------------
# Typeinfer: rank-N input + dtype-mismatch (x bf16 / weight f32) accepted;
# output keeps x's full shape and dtype. Rank-0 x / rank-2 weight / last-dim
# mismatch rejected.
# ---------------------------------------------------------------------------

_CTX_LEN = DimVar("CTX_LEN", 1, 4097)
_RMS = RMSNorm(eps=1e-6)

CASES = [
    TypeInferCase(
        "rank3_bf16_input_f32_weight",
        _RMS,
        (_ten((1, 1, 2048), DType.bf16), _ten((2048,), DType.f32)),
        _ten((1, 1, 2048), DType.bf16),
    ),
    # dynamic batch dim (DimVar arithmetic) flows through verbatim.
    TypeInferCase(
        "dim_arithmetic_batch_survives",
        _RMS,
        (_ten((1, _CTX_LEN + 1, 2048), DType.bf16), _ten((2048,), DType.f32)),
        _ten((1, _CTX_LEN + 1, 2048), DType.bf16),
    ),
    TypeInferCase(
        "rank2_same_dtype",
        _RMS,
        (_ten((4, 2048), DType.bf16), _ten((2048,), DType.bf16)),
        _ten((4, 2048), DType.bf16),
    ),
    # dtype mismatch (bf16 x / f32 weight) is legal; output keeps x's dtype.
    TypeInferCase(
        "rank2_dtype_mismatch_allowed",
        _RMS,
        (_ten((4, 2048), DType.bf16), _ten((2048,), DType.f32)),
        _ten((4, 2048), DType.bf16),
    ),
    TypeInferCase(
        "rank0_x_rejected",
        _RMS,
        (TensorType.scalar(DType.bf16), _ten((2048,), DType.f32)),
        ExpectedError(match="x must be rank ≥ 1", exc=TypeError),
    ),
    TypeInferCase(
        "rank2_weight_rejected",
        _RMS,
        (_ten((4, 2048), DType.bf16), _ten((1, 2048), DType.f32)),
        ExpectedError(match="weight must be rank-1", exc=TypeError),
    ),
    TypeInferCase(
        "last_dim_mismatch_rejected",
        _RMS,
        (_ten((4, 2048), DType.bf16), _ten((1024,), DType.f32)),
        ExpectedError(match="last dim", exc=TypeError),
    ),
    # rms_norm normalizes across an axis (non-monotonic); no reduction commutes.
    TypeInferCase(
        "partial_input_rejected",
        _RMS,
        (
            sharded((4, 2048), (Partial("sum"),), mesh((4,)), dtype=DType.bf16),
            _ten((2048,), DType.f32),
        ),
        ExpectedError(match="Partial input on x is unsound", exc=TypeError),
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_rms_norm_typeinfer(case):
    run_typeinfer_case(case)


# ===========================================================================
# HIR->TIR lowering / CUDA codegen (verbatim from test_qwen3_rmsnorm.py).
# ===========================================================================


# kinded Binary/Unary math ops are built via helpers, not per-name classes.
# These local helper constructors keep the IR-construction tests readable
# without re-introducing the legacy class names — every call site lifts to a
# fresh ``Binary(kind=...)``.
def Add() -> Binary:
    return Binary(kind=BinaryKind.ADD)


def Mul() -> Binary:
    return Binary(kind=BinaryKind.MUL)


def rsqrt_op() -> Unary:
    return Unary(kind=UnaryKind.RSQRT)


_DEFAULT_DTYPE = DType.bf16
_COMPUTE_DTYPE = DType.f32


def _make_meshes():
    cta = Topology("cta", 128)
    thread = Topology("thread", 256)
    cta_mesh = Mesh(topology=cta, layout=Layout(shape=(128,), strides=(1,)))
    thread_mesh = Mesh(topology=thread, layout=Layout(shape=(256,), strides=(1,)))
    return cta_mesh, thread_mesh


# ── Builder A: declarative reduce-op RMSNorm ────────────────────────


def _build_ssa_rmsnorm(M, K, dtype=_DEFAULT_DTYPE, compute_dtype=_COMPUTE_DTYPE):
    inp_type = TensorType(shape=(M, K), dtype=dtype, layout=None, storage="gmem")
    w_type = TensorType(shape=(K,), dtype=dtype, layout=None, storage="gmem")
    x = Var(type=inp_type, name="x")
    weight = Var(type=w_type, name="weight")
    eps = 1e-6

    f32_type = TensorType(shape=(M, K), dtype=compute_dtype, layout=None, storage="gmem")
    x_f32 = Call(type=f32_type, target=Cast(dtype=compute_dtype), args=(x,))
    x_sq = Call(type=f32_type, target=Mul(), args=(x_f32, x_f32))
    mean_type = TensorType(shape=(M, 1), dtype=compute_dtype, layout=None, storage="gmem")
    mean = Call(type=mean_type, target=Reduce(axes=(-1,), keepdim=True, kind=ReduceKind.MEAN), args=(x_sq,))
    eps_const = Constant(value=eps, type=TensorType(shape=(), dtype=compute_dtype, layout=None, storage="gmem"))
    ms_eps = Call(type=mean_type, target=Add(), args=(mean, eps_const))
    rms = Call(type=mean_type, target=rsqrt_op(), args=(ms_eps,))
    x_norm = Call(type=f32_type, target=Mul(), args=(x_f32, rms))
    w_f32_type = TensorType(shape=(K,), dtype=compute_dtype, layout=None, storage="gmem")
    w_f32 = Call(type=w_f32_type, target=Cast(dtype=compute_dtype), args=(weight,))
    y_f32 = Call(type=f32_type, target=Mul(), args=(x_norm, w_f32))
    y = Call(type=inp_type, target=Cast(dtype=dtype), args=(y_f32,))
    return Function.build(name="rmsnorm_ssa", params=(x, weight), body=y, return_type=inp_type)


# ── Builder B: GridRegion per-row RMSNorm ────────────────────────────


def _build_gridregion_rmsnorm(M, K, dtype=_DEFAULT_DTYPE, compute_dtype=_COMPUTE_DTYPE):

    inp_type = TensorType(shape=(M, K), dtype=dtype, layout=None, storage="gmem")
    w_type = TensorType(shape=(K,), dtype=dtype, layout=None, storage="gmem")
    x = Var(type=inp_type, name="x")
    weight = Var(type=w_type, name="weight")
    eps = 1e-6

    iv = Var(type=TensorType(shape=(), dtype=DType.i32, layout=None, storage="rmem"), name="m")
    row_type = TensorType(shape=(K,), dtype=dtype, layout=None, storage="gmem")
    row_f32_type = TensorType(shape=(K,), dtype=compute_dtype, layout=None, storage="rmem")

    row_x = Call(type=row_type, target=Gather(axis=0), args=(x, iv))
    row_f32 = Call(type=row_f32_type, target=Cast(dtype=compute_dtype), args=(row_x,))
    row_sq = Call(type=row_f32_type, target=Mul(), args=(row_f32, row_f32))
    scalar_type = TensorType(shape=(), dtype=compute_dtype, layout=None, storage="rmem")
    mean = Call(type=scalar_type, target=Reduce(axes=(0,), keepdim=True, kind=ReduceKind.MEAN), args=(row_sq,))
    eps_const = Constant(value=eps, type=TensorType(shape=(), dtype=compute_dtype, layout=None, storage="rmem"))
    ms_eps = Call(type=scalar_type, target=Add(), args=(mean, eps_const))
    rms = Call(type=scalar_type, target=rsqrt_op(), args=(ms_eps,))
    row_norm = Call(type=row_f32_type, target=Mul(), args=(row_f32, rms))
    w_f32_type = TensorType(shape=(K,), dtype=compute_dtype, layout=None, storage="gmem")
    w_f32 = Call(type=w_f32_type, target=Cast(dtype=compute_dtype), args=(weight,))
    row_out_f32 = Call(type=row_f32_type, target=Mul(), args=(row_norm, w_f32))
    row_out = Call(type=row_type, target=Cast(dtype=dtype), args=(row_out_f32,))

    grid = GridRegionExpr(
        type=inp_type,
        induction_var=iv,
        carried_args=(),
        init_args=(),
        body=row_out,
        yield_values=(),
        extent=M,
        step=1,
    )
    return Function.build(name="rmsnorm_grid", params=(x, weight), body=grid, return_type=inp_type)


# ── tests ───────────────────────────────────────────────────────────


class TestSSARMSNorm:
    def test_typeinfer_output_shape_and_dtype(self):
        fn = _build_ssa_rmsnorm(M=1, K=2048)
        assert fn.body.type.shape == (1, 2048)
        assert fn.body.type.dtype == _DEFAULT_DTYPE

    def test_lowering_produces_tir_with_reduce(self):
        # noqa PLC0415: shadows the HIR Binary/Reduce that this file uses at module scope.
        from tilefoundry.ir.tir.arith import Binary, BinaryKind, Unary, UnaryKind  # noqa: PLC0415
        from tilefoundry.ir.tir.reduce import Reduce  # noqa: PLC0415

        fn = _build_ssa_rmsnorm(M=1, K=2048)
        cta_mesh, thread_mesh = _make_meshes()
        mod = Module(name="t", functions=(fn,), entry=fn.name)
        result = HirToTirPass(_cta=cta_mesh, _thread=thread_mesh).run(mod)

        def _is_eval_of(stmt, op_cls) -> bool:
            return isinstance(stmt, Evaluate) and isinstance(stmt.callable, op_cls)

        prim = result.functions[0]
        assert isinstance(prim, PrimFunction)
        stmts = list(_walk_stmts(prim.body))
        type_names = [type(s).__name__ for s in stmts]
        # Reduce is an Op invoked as Evaluate(Reduce, ...)
        assert any(_is_eval_of(s, Reduce) for s in stmts), f"expected Evaluate(Reduce), got {type_names}"
        assert any(_is_eval_of(s, Unary) and s.callable.kind == UnaryKind.RSQRT for s in stmts), f"expected Evaluate(Unary(RSQRT)), got {type_names}"
        assert any(_is_eval_of(s, Binary) and s.callable.kind == BinaryKind.MUL for s in stmts), f"expected Evaluate(Binary(MUL)), got {type_names}"


class TestGridRegionRMSNorm:
    def test_typeinfer_output_shape_and_dtype(self):
        fn = _build_gridregion_rmsnorm(M=1, K=2048)
        assert fn.return_type.shape == (1, 2048)
        assert fn.return_type.dtype == _DEFAULT_DTYPE

    def test_lowering_produces_for_loop_and_reduce(self):
        # noqa PLC0415: shadows the HIR Reduce that this file uses at module scope.
        from tilefoundry.ir.tir.reduce import Reduce  # noqa: PLC0415

        fn = _build_gridregion_rmsnorm(M=2, K=4)
        cta_mesh, thread_mesh = _make_meshes()
        mod = Module(name="t", functions=(fn,), entry=fn.name)
        result = HirToTirPass(_cta=cta_mesh, _thread=thread_mesh).run(mod)

        prim = result.functions[0]
        assert isinstance(prim, PrimFunction)
        stmts = list(_walk_stmts(prim.body))
        type_names = [type(s).__name__ for s in stmts]
        assert any(isinstance(s, For) for s in stmts), f"expected For loop, got {type_names}"
        # Reduce is an Op invoked as Evaluate(Reduce, ...)
        assert any(
            isinstance(s, Evaluate) and isinstance(s.callable, Reduce) for s in stmts
        ), f"expected Evaluate(Reduce), got {type_names}"


def _walk_stmts(stmt):
    if isinstance(stmt, Sequential):
        for s in stmt.body:
            yield from _walk_stmts(s)
    elif isinstance(stmt, LetStmt):
        yield stmt
        yield from _walk_stmts(stmt.body)
    elif isinstance(stmt, MeshScope):
        yield stmt
        yield from _walk_stmts(stmt.body)
    elif isinstance(stmt, (For, While)):
        yield stmt
        yield from _walk_stmts(stmt.body)
    elif isinstance(stmt, If):
        yield stmt
        yield from _walk_stmts(stmt.then_body)
        yield from _walk_stmts(stmt.else_body)
    elif isinstance(stmt, Stmt):
        yield stmt


def test_rms_norm_evaluate():
    torch.manual_seed(0)
    _nx, _nw = torch.randn(2, 8), torch.randn(8)
    _nref = _nx * torch.rsqrt(_nx.pow(2).mean(-1, keepdim=True) + 1e-6) * _nw
    run_eval_case(EvalCase("", RMSNorm(eps=1e-6), (_nx, _nw), _nref, atol=1e-5))
