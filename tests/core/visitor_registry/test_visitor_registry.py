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

from tilefoundry.interfaces import cost, evaluator, typeinfer, verify
from tilefoundry.ir import core, expr, types
from tilefoundry.ir.tir import memory


def _t() -> types.TensorType:
    return types.TensorType.from_scalar(expr.DType.f32)


def _is_builtin_hir_op(op_class: type[expr.Op]) -> bool:
    parts = op_class.__module__.split(".")
    return len(parts) >= 5 and parts[:3] == ["tilefoundry", "ir", "hir"]


def test_every_real_op_has_typeinfer_value_and_cost() -> None:
    """Report every builtin HIR Op whose analysis registries are incomplete."""
    schemas = [
        schema
        for schema in expr.iter_schemas()
        if not schema.is_alias and _is_builtin_hir_op(schema.op_class)
    ]
    registries = (typeinfer.registry, evaluator.registry, cost.registry)
    missing = {
        schema.name: [registry.name for registry in registries if not registry.has(schema.op_class)]
        for schema in schemas
    }

    assert {name: gaps for name, gaps in missing.items() if gaps} == {}


_THREAD = types.Topology("thread", 32)


_BUFFER = types.ShardLayout(
    types.Layout((8,), (1,)), (types.Broadcast(),), types.Mesh((_THREAD,), types.Layout((32,), (1,)), ("t",))
)


def _copied(shape: tuple, layout=None) -> types.TensorType:
    return types.TensorType(shape=shape, dtype=expr.DType.f32, layout=layout, storage="rmem")


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        pytest.param(_copied((4,)), _copied((8,)), id="plain_shapes_differ"),
        pytest.param(
            _copied((2, 4), _BUFFER), _copied((8,), _BUFFER),
            id="one_buffer_rows_onto_a_line",
        ),
        pytest.param(
            _copied((8,), _BUFFER), _copied((2, 4), _BUFFER),
            id="one_buffer_a_line_onto_rows",
        ),
    ],
)
def test_verify_visitor_copy_evaluate_dispatch_and_unregistered_passthrough(src, dst) -> None:
    """Copy's declared SameShape rejects reshapes, including one per-thread buffer.

    The execution scope is valid, so rejection must come from the declaration,
    rather than an absent leaf verifier or an access-relation projection error.
    Structural Return and LetStmt still traverse without a custom verifier.
    """
    src_var, dst_var = expr.Var(type=src, name="src"), expr.Var(type=dst, name="dst")
    stmt = core.Evaluate(callable=memory.Copy(), args=(src_var, dst_var))
    ctx = verify.Context(current_mesh=_BUFFER.mesh, mesh_scope=(_BUFFER.mesh,))
    with pytest.raises(expr.VerifyError, match=r"reads src.shape = dst.shape"):
        verify.Visitor(ctx).visit(stmt)

    verify.Visitor(verify.Context()).visit(core.Return())
    verify.Visitor(verify.Context()).visit(
        core.LetStmt(
            var=expr.Var(type=_t(), name="x"),
            value=expr.Constant(type=_t(), value=1.0),
        )
    )
