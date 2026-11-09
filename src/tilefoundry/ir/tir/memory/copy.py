"""Effect-ful TIR Op ``tir.memory.Copy``.

Copies ``src`` to ``dst``. Memory direction (gmem / smem
/ rmem) is inferred from ``.type.storage`` of each operand. Covers
Load / Store too. The Op is placed in Stmt position as
``Evaluate(Copy, ...)``; the invocation is unit-typed (no result value).
"""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, evaluator, typeinfer
from tilefoundry.interfaces import verify as interfaces_verify
from tilefoundry.ir import expr, pattern, types


@expr.register_op(dialect='T', category='memory')
class Copy(expr.Op):
    """Copies ``src`` into ``dst`` (in-place memory write)."""

    capability = expr.OpCapability(None)

    execution_mesh = pattern.memory_execution_mesh()

    src = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ,
        pattern=pattern.operand_tile(0, execution_mesh=execution_mesh),
    )
    dst = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.WRITE,
        pattern=pattern.operand_tile(1, execution_mesh=execution_mesh),
    )

    between = (pattern.SameConstraint("shape", "src", "dst"), pattern.SameConstraint("dtype", "src", "dst"))


@typeinfer.register(Copy)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


access.register_relation(Copy)(access.identity_relations)


@evaluator.register(Copy)
def eval_scheduled_copy(ctx):
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, ctx.args[0].data))


def register_copy_semantics(op, *, verify=True):
    """Reuse Copy's value, type and access rules for a transfer declaration."""
    typeinfer.register(op)(typeinfer.registry.require(Copy))
    access.register_relation(op)(access.identity_relations)
    evaluator.register(op)(evaluator.registry.require(Copy))
    rule = interfaces_verify.registry.get(Copy, None)
    if verify and rule is not None:
        interfaces_verify.register(op)(rule)
    return op


__all__ = ["Copy", "register_copy_semantics"]
