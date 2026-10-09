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

import isl
import pytest

import tilefoundry
from tilefoundry.evaluator import eval_registry
from tilefoundry.ir.core import Call, Constant, Op, Var
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.ir.core.op_registry import iter_schemas
from tilefoundry.ir.tir.memory import Copy
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, Return, Sequential
from tilefoundry.ir.types import (
    DType,
    Layout,
    Mesh,
    ShardLayout,
    TensorType,
    Topology,
    UnitType,
)
from tilefoundry.ir.types.shard_layout import Broadcast
from tilefoundry.target import CudaTarget
from tilefoundry.visitor_registry.access_relation import (
    local_relations_of,
    reached_elements,
    relations_of,
)
from tilefoundry.visitor_registry.contexts import (
    CostContext,
    FunctionScope,
    TypeInferContext,
    VerifyContext,
)
from tilefoundry.visitor_registry.registries import (
    codegen_registry,
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


_THREAD = Topology("thread", 32)
_BUFFER = ShardLayout(
    Layout((8,), (1,)), (Broadcast(),), Mesh((_THREAD,), Layout((32,), (1,)), ("t",))
)


def _copied(shape: tuple, layout=None) -> TensorType:
    return TensorType(shape=shape, dtype=DType.f32, layout=layout, storage="rmem")


@pytest.mark.parametrize(
    ("src", "dst", "reached"),
    [
        pytest.param(_copied((4,)), _copied((8,)), None, id="plain_shapes_differ"),
        pytest.param(
            _copied((2, 4), _BUFFER),
            _copied((8,), _BUFFER),
            "{ [d0, d1] -> [4d0 + d1] : 0 <= d0 < 2 and 0 <= d1 < 4 }",
            id="one_buffer_rows_onto_a_line",
        ),
        pytest.param(
            _copied((8,), _BUFFER),
            _copied((2, 4), _BUFFER),
            "{ [d0] -> [floor(d0/4), d0 mod 4] : 0 <= d0 < 8 }",
            id="one_buffer_a_line_onto_rows",
        ),
    ],
)
def test_verify_visitor_copy_evaluate_dispatch_and_unregistered_passthrough(
    src, dst, reached
) -> None:
    """``Evaluate(Copy, ...)`` dispatches verify on Op class.

    A Copy between shapes that differ is refused unless both describe one
    per-thread buffer; then ``dst`` is reached where the same buffer position
    holds it, the row-major reshape of ``src``'s coordinates, and each thread
    reads and writes that whole buffer once. Unregistered structural Stmts
    (Return / LetStmt) pass through silently.
    """
    src_var, dst_var = Var(type=src, name="src"), Var(type=dst, name="dst")
    stmt = Evaluate(callable=Copy(), args=(src_var, dst_var))

    if reached is None:
        with pytest.raises(VerifyError, match=r"^Copy: "):
            VerifyVisitor(VerifyContext()).visit(stmt)
    else:
        VerifyVisitor(VerifyContext()).visit(stmt)
        call = Call(type=UnitType(), target=Copy(), args=(src_var, dst_var))
        written = relations_of(call, CostContext())[1]
        assert written.relation.is_equal(isl.map(reached))
        held = local_relations_of(call, CostContext(topology_level="thread", topologies=(_THREAD,)))
        operands = held[: len(call.args)]
        read_at, written_at = (boundary.relation for boundary in operands)
        assert read_at.is_equal(written_at), "each iteration reads and writes one position"
        assert written_at.range().is_equal(isl.set("{ [p] : 0 <= p < 8 }"))
        assert [reached_elements(boundary) * 4 for boundary in operands] == [32, 32]
        assert all(not boundary.values for boundary in held)

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
    miss = r"codegen: nothing registered for \(CudaTarget, Role.EMIT, _UnknownOp\)"
    with pytest.raises(RuntimeError, match=miss):
        CodegenVisitor(_Ctx(), codegen_registry, target=CudaTarget).emit_expr(call)
    with pytest.raises(VerifyError, match="no cost evaluator registered for _UnknownOp"):
        CostEvaluator().visit_Call(call, CostContext())


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
        assert declared[:3] == ["scope", "current_mesh", "memo"]
        assert not any(
            hasattr(context, name)
            for name in ("module", "caller", "child_call", "child_call_owner")
        )
