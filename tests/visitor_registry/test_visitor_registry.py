"""``tilefoundry.visitor_registry`` — dispatch on the Op class.

``tilefoundry.visitor_registry`` — dispatch on the Op class, and what happens
when nothing is registered for it.

Every model run dispatches thousands of registered visits, so the positive path
needs no separate witness. What a model cannot show is the shape of the *miss*:
an unregistered structural Stmt must pass through, while an unregistered Op must
raise rather than return a zero.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

import tilefoundry
from tilefoundry.evaluator import eval_registry
from tilefoundry.ir.core import Call, Constant, Op, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, Return, Sequential
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry.contexts import (
    CostContext,
    FunctionScope,
    TypeInferContext,
    VerifyContext,
)
from tilefoundry.visitor_registry.registries import (
    codegen_cuda_registry,
    cost_evaluator_registry,
    typeinfer_registry,
)
from tilefoundry.visitor_registry.visitors import (
    CodegenVisitor,
    CostEvaluator,
    VerifyVisitor,
)


def _t() -> TensorType:
    return TensorType.scalar(DType.f32)


EXPECTED_BUILTIN_HIR_OP_NAMES = {
    "argmax",
    "binary",
    "cache_update",
    "cast",
    "clamp",
    "concat",
    "conv2d",
    "full_like",
    "gather",
    "gelu",
    "insert_slice",
    "layer_norm",
    "local",
    "matmul",
    "mma_sm80_16x8x16",
    "quant",
    "rank",
    "reduce",
    "relu",
    "repeat_interleave",
    "reshape",
    "reshard",
    "rms_norm",
    "rope",
    "shape_compose",
    "shape_extract",
    "shape_of",
    "sigmoid",
    "silu",
    "slice",
    "softmax",
    "softplus",
    "split",
    "stack",
    "tanh",
    "topk",
    "transpose",
    "tuple_get_item",
    "unary",
    "wgmma_sm90_64x128x16",
    "zeros",
}

EXPECTED_MISSING = {}


def _is_builtin_hir_op(op_class: type[Op]) -> bool:
    parts = op_class.__module__.split(".")
    return len(parts) >= 5 and parts[:3] == ["tilefoundry", "ir", "hir"]


def test_every_real_op_has_typeinfer_value_and_cost() -> None:
    """Report every builtin HIR Op whose analysis registries are incomplete."""
    schemas = [
        schema
        for schema in iter_schemas()
        if not schema.is_alias and _is_builtin_hir_op(schema.op_class)
    ]
    assert len(schemas) == 41
    assert {schema.name for schema in schemas} == EXPECTED_BUILTIN_HIR_OP_NAMES

    registries = (typeinfer_registry, eval_registry, cost_evaluator_registry)
    missing = {
        schema.name: [registry.name for registry in registries if not registry.has(schema.op_class)]
        for schema in schemas
    }

    assert {name: gaps for name, gaps in missing.items() if gaps} == EXPECTED_MISSING


def test_verify_visitor_copy_evaluate_dispatch_and_unregistered_passthrough() -> None:
    """``Evaluate(Copy, ...)`` dispatches verify on Op class.

    ``Evaluate(Copy, ...)`` dispatches verify on Op class;
    unregistered structural Stmts (Return / LetStmt) pass through silently.
    """
    src = Var(type=TensorType(shape=(4,), dtype=DType.f32, layout=None, storage="rmem"), name="src")
    dst = Var(type=TensorType(shape=(8,), dtype=DType.f32, layout=None, storage="rmem"), name="dst")
    stmt = Evaluate(callable=Copy(), args=(src, dst))

    ctx = VerifyContext()
    with pytest.raises(VerifyError, match=r"^Copy: "):
        VerifyVisitor(ctx).visit(stmt)

    VerifyVisitor(VerifyContext()).visit(Return())
    VerifyVisitor(VerifyContext()).visit(
        LetStmt(
            var=Var(type=_t(), name="x"),
            value=Constant(type=_t(), value=1.0),
            body=Sequential(body=()),
        )
    )


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
    with pytest.raises(RuntimeError, match="no @register_codegen_cuda for Op _UnknownOp"):
        CodegenVisitor(_Ctx(), codegen_cuda_registry, backend="cuda").emit_expr(call)
    with pytest.raises(VerifyError, match="no cost evaluator registered for _UnknownOp"):
        CostEvaluator(CostContext()).visit_Call(call)


def test_where_a_walk_reads_is_one_pair_and_nothing_else() -> None:
    """The location API is `FunctionScope` and `TypeInferContext.scope`.

    Both are reachable from the package root, because one is how the other is
    constructed. Nothing else on a context describes where a walk reads, and no
    context answers a question about one kind of construct.
    """
    assert (tilefoundry.FunctionScope, tilefoundry.TypeInferContext) == (
        FunctionScope,
        TypeInferContext,
    )
    assert FunctionScope.__dataclass_params__.frozen
    assert [field.name for field in fields(FunctionScope)] == ["module", "function"]
    assert [field.type for field in fields(FunctionScope)] == ["Module", "Function"]

    for context in (TypeInferContext, VerifyContext, CostContext):
        declared = [field.name for field in fields(context)]
        assert declared[:4] == ["scope", "cache", "mesh_scope", "elaboration_cache"]
        assert not any(
            hasattr(context, name)
            for name in ("module", "caller", "child_call", "child_call_owner")
        )
