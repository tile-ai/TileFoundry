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
from tilefoundry.analysis.compute_cost import _prove_storage, _Storage
from tilefoundry.evaluator import eval_registry
from tilefoundry.ir.core import Call, Constant, Op, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, Return, Sequential
from tilefoundry.ir.types import DType, TensorType, tensor_bytes
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.access_relation import (
    StorageClaim,
    StorageEffect,
    StorageSpan,
    identity_relations,
    register_access_relation,
)
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
    registries = (typeinfer_registry, eval_registry, cost_evaluator_registry)
    missing = {
        schema.name: [registry.name for registry in registries if not registry.has(schema.op_class)]
        for schema in schemas
    }

    assert {name: gaps for name, gaps in missing.items() if gaps} == {}


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


def test_a_storage_claim_covers_every_operand_it_names() -> None:
    """Addressing one operand does not conclude anything about a second.

    A conclusion is read as "the result lives in these operands", and its reader
    retires the movement of each. A handler that names two while proving one
    would have the reader retire bytes nothing was shown about, so the claim is
    refused rather than trimmed to the part that held. An Op with no boundary
    relation still states this: the two are one registration, not one fact.
    """

    class _ReachesBoth(Op):
        pass

    class _ReachesOne(Op):
        pass

    holds = TensorType(shape=(4,), dtype=DType.f32, layout=None, storage=StorageKind.GMEM)
    size = tensor_bytes(holds)
    left = Var(type=holds, name="left")
    right = Var(type=holds, name="right")

    def _both(call: Call, ctx) -> StorageClaim:
        return StorageClaim(StorageEffect.FORWARD, (0,), (StorageSpan(0, 0, size),))

    def _one(call: Call, ctx) -> StorageClaim:
        return StorageClaim(StorageEffect.FORWARD, (0, 1), (StorageSpan(0, 0, size),))

    register_access_relation(_ReachesBoth)(identity_relations(2, _both))
    register_access_relation(_ReachesOne)(identity_relations(2, _one))

    walk = _Storage(
        type_of=lambda expr: expr.type, users={}, positions={}, caller_owned=frozenset()
    )
    covered = Call(type=holds, target=_ReachesBoth(), args=(left, right))
    assert _prove_storage(covered, walk) == (StorageEffect.FORWARD, (0,))

    partial = Call(type=holds, target=_ReachesOne(), args=(left, right))
    assert _prove_storage(partial, walk) is None
    assert id(partial) not in walk.bases


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
