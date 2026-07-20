"""CUDA emitter for ``tir.If`` + scalar predicate path.

Two concerns, each with one positive and one negative test:

1. ``render_scalar_predicate`` — Constants / Vars / supported scalar
   binary kinds (positive); unsupported Expr kind rejected (negative).
2. ``tir.If`` emitter — non-empty else body emits ``else { ... }``;
   empty else body collapses to a single ``if (...) { ... }``.
"""

from __future__ import annotations

import tilefoundry.codegen.cuda  # noqa: F401  — trigger emitter autodiscovery
from tilefoundry.codegen.cuda.context import CodegenContext
from tilefoundry.codegen.cuda.tir.stmts.if_ import render_scalar_predicate
from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.tir.stmts import If, Sequential
from tilefoundry.ir.types import DType, TensorType


def _const(value, *, dtype: DType = DType.i32) -> Constant:
    return Constant(value=value, type=TensorType.scalar(dtype=dtype))


def _var(name: str, *, dtype: DType = DType.i32) -> Var:
    return Var(name=name, type=TensorType.scalar(dtype=dtype))


def _scalar_binary(kind: BinaryKind, lhs, rhs, *, result_dtype: DType) -> Call:
    return Call(
        target=HirBinary(kind=kind),
        args=(lhs, rhs),
        type=TensorType.scalar(dtype=result_dtype),
    )


# --- 1. scalar predicate renderer ---------------------------------------


def test_render_scalar_predicate_covers_supported_forms() -> None:
    """``1 <= S && S < 4`` exercises Constant + Var + comparison + AND."""
    ctx = CodegenContext()
    s = _var("S")
    lo = _scalar_binary(BinaryKind.LE, _const(1), s, result_dtype=DType.bool)
    hi = _scalar_binary(BinaryKind.LT, s, _const(4), result_dtype=DType.bool)
    expr = _scalar_binary(BinaryKind.AND, lo, hi, result_dtype=DType.bool)
    assert (
        render_scalar_predicate(expr, ctx)
        == "((1) <= (S_1)) && ((S_1) < (4))"
    )


# --- 2. If emitter -------------------------------------------------------


def _emit_to_source(node) -> str:
    ctx = CodegenContext()
    ctx.emit_node(node)
    return ctx.source()


def test_if_emitter_renders_then_and_else_blocks() -> None:
    """Non-empty else body produces ``else { ... }``; cond uses the scalar renderer."""
    cond = _scalar_binary(
        BinaryKind.LT, _var("S"), _const(4), result_dtype=DType.bool
    )
    body = Sequential(body=(Sequential(body=()),))  # any non-empty Sequential
    src = _emit_to_source(If(cond=cond, then_body=body, else_body=body))
    assert "if ((S_1) < (4)) {" in src
    assert "} else {" in src
    assert src.rstrip().endswith("}")


def test_if_emitter_skips_else_when_else_body_empty() -> None:
    """Empty else body collapses to a single ``if (...) { ... }``."""
    src = _emit_to_source(
        If(
            cond=_const(True, dtype=DType.bool),
            then_body=Sequential(body=()),
            else_body=Sequential(body=()),
        )
    )
    assert "if (true) {" in src
    assert "else" not in src
