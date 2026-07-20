"""``tilefoundry.visitor_registry`` — registry contract + canonical visitors."""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Call, Constant, Op, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.hir.math.binary import Binary
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, Return, Sequential
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import typeinfer_registry
from tilefoundry.visitor_registry.contexts import (
    CostContext,
    TypeInferContext,
    VerifyContext,
)
from tilefoundry.visitor_registry.visitors import (
    CodegenVisitor,
    CostEvaluator,
    TypeInferVisitor,
    VerifyVisitor,
)


def _t() -> TensorType:
    return TensorType.scalar(DType.f32)


def test_typeinfer_visitor_dispatches_through_canonical_registry() -> None:
    """``import tilefoundry.ir.hir`` populates ``typeinfer_registry``;
    visitor dispatches Call → registered handler."""

    assert typeinfer_registry.has(Binary)

    a = Var(type=_t(), name="a")
    b = Var(type=_t(), name="b")
    out = TypeInferVisitor(TypeInferContext()).visit(
        Call(type=_t(), target=Binary(kind=BinaryKind.ADD), args=(a, b))
    )
    assert out == _t()


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
        CodegenVisitor(_Ctx(), target="cuda").emit_expr(call)
    with pytest.raises(VerifyError, match="no cost evaluator registered for _UnknownOp"):
        CostEvaluator(CostContext()).visit_Call(call)
