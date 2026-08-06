"""``tilefoundry.visitor_registry`` — dispatch on the Op class, and what happens
when nothing is registered for it.

Every model run dispatches thousands of registered visits, so the positive path
needs no separate witness. What a model cannot show is the shape of the *miss*:
an unregistered structural Stmt must pass through, while an unregistered Op must
raise rather than return a zero.
"""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Call, Constant, Op, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, Return, Sequential
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry.contexts import CostContext, VerifyContext
from tilefoundry.visitor_registry.registries import codegen_cuda_registry
from tilefoundry.visitor_registry.visitors import (
    CodegenVisitor,
    CostEvaluator,
    VerifyVisitor,
)


def _t() -> TensorType:
    return TensorType.scalar(DType.f32)


def test_verify_visitor_copy_evaluate_dispatch_and_unregistered_passthrough() -> None:
    """``Evaluate(Copy, ...)`` dispatches verify on Op class;
    unregistered structural Stmts (Return / LetStmt) pass through silently."""

    src = Var(type=TensorType(shape=(4,), dtype=DType.f32, layout=None, storage="rmem"), name="src")
    dst = Var(type=TensorType(shape=(8,), dtype=DType.f32, layout=None, storage="rmem"), name="dst")
    stmt = Evaluate(callable=Copy(), args=(src, dst))

    ctx = VerifyContext()
    with pytest.raises(VerifyError, match=r"^Copy: "):
        VerifyVisitor(ctx).visit(stmt)

    # Unregistered structural Stmts are no-ops.
    VerifyVisitor(VerifyContext()).visit(Return())
    VerifyVisitor(VerifyContext()).visit(
        LetStmt(
            var=Var(type=_t(), name="x"),
            value=Constant(type=_t(), value=1.0),
            body=Sequential(body=()),
        )
    )


def test_visitors_fail_closed_when_unregistered() -> None:
    """An Op with no registered handler is an error, never a silent no-op
    or a zero result — for codegen and Cost Evaluators alike."""
    class _UnknownOp(Op):
        pass

    class _Ctx:
        pass

    call = Call(type=_t(), target=_UnknownOp(), args=())
    with pytest.raises(RuntimeError, match="no @register_codegen_cuda for Op _UnknownOp"):
        CodegenVisitor(
            _Ctx(), codegen_cuda_registry, backend="cuda"
        ).emit_expr(call)
    with pytest.raises(VerifyError, match="no cost evaluator registered for _UnknownOp"):
        CostEvaluator(CostContext()).visit_Call(call)
