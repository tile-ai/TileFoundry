"""Closed failure of missing CUDA operation visitors."""

from __future__ import annotations

import pytest

from tests.core.visitor_registry.test_visitor_registry import (
    _t,
)
from tilefoundry import codegen
from tilefoundry.interfaces import cost
from tilefoundry.ir import expr
from tilefoundry_nvidia import cuda


def test_visitors_fail_closed_when_unregistered() -> None:
    """An Op with no registered handler is an error, never a silent no-op or a zero result.

    An Op with no registered handler is an error, never a silent no-op
    or a zero result — for codegen and Cost Evaluators alike.
    """

    class _UnknownOp(expr.Op):
        pass

    class _Ctx:
        pass

    call = expr.Call(type=_t(), target=_UnknownOp(), args=())
    miss = r"codegen: nothing registered for \(cuda.Target, Role.EMIT, _UnknownOp\)"
    with pytest.raises(RuntimeError, match=miss):
        codegen.Visitor(_Ctx(), codegen.registry, target=cuda.Target).emit_expr(call)
    with pytest.raises(expr.VerifyError, match="no cost evaluator registered for _UnknownOp"):
        cost.Visitor().visit_Call(call, cost.Context())
