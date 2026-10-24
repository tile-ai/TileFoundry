"""``tilefoundry.visitor_registry`` — dispatch on the Op class.

``tilefoundry.visitor_registry`` — dispatch on the Op class, and what happens
when nothing is registered for it.

Every model run dispatches thousands of registered visits, so the positive path
needs no separate witness. What a model cannot show is the shape of the *miss*:
an unregistered structural Stmt must pass through, while an unregistered Op must
raise rather than return a zero.
"""

from __future__ import annotations

import pytest

from tests.core.visitor_registry.test_visitor_registry import (
    _t,
)
from tilefoundry.ir.core import Call, Op
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.visitor_registry.contexts import CostContext
from tilefoundry.visitor_registry.registries import (
    codegen_registry,
)
from tilefoundry.visitor_registry.visitors import (
    CodegenVisitor,
    CostEvaluator,
)
from tilefoundry_nvidia import CudaTarget


def test_visitors_fail_closed_when_unregistered() -> None:
    """An Op with no registered handler is an error, never a silent no-op or a zero result.

    An Op with no registered handler is an error, never a silent no-op
    or a zero result — for codegen and Cost Evaluators alike.
    """

    class _UnknownOp(Op):
        pass

    class _Ctx:
        pass

    call = Call(type=_t(), target=_UnknownOp(), args=())
    miss = r"codegen: nothing registered for \(CudaTarget, Role.EMIT, _UnknownOp\)"
    with pytest.raises(RuntimeError, match=miss):
        CodegenVisitor(_Ctx(), codegen_registry, target=CudaTarget).emit_expr(call)
    with pytest.raises(VerifyError, match="no cost evaluator registered for _UnknownOp"):
        CostEvaluator().visit_Call(call, CostContext())
