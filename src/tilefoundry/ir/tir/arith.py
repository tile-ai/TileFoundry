"""Generic TIR effect-form Ops for binary and unary pointwise operations.

Dispatched by kind enum. ``BinaryKind`` and ``UnaryKind`` are owned by
``tilefoundry.ir.expr.kinds`` so HIR and TIR carry the same enum values
without remapping.

``Binary`` / ``Unary`` are ``Op`` subclasses (not ``Stmt``); in Stmt
position they are invoked as ``Evaluate(op, args)`` (unit-typed, no
result value), matching the TIR convention shared with
``tir.memory.Copy``.
"""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, evaluator, typeinfer, verify
from tilefoundry.ir import expr, pattern, types

__all__ = ["Binary", "Unary"]

IN_RMEM = pattern.tensor_in(expr.StorageKind.RMEM)


@expr.register_op(dialect="T", category="arith")
class Binary(expr.Op):
    """Effect-form pointwise binary operation: ``dst = lhs <kind> rhs``."""

    capability = expr.OpCapability(None)
    execution_mesh = pattern.thread_execution_mesh()

    lhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=IN_RMEM)
    rhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=IN_RMEM)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=IN_RMEM)
    kind = expr.ParamDef(kind="attribute", annotation=expr.BinaryKind)


@typeinfer.register(Binary)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


access.register_relation(Binary)(access.identity_relations)


@evaluator.register(Binary)
def eval_scheduled_binary(ctx):
    return evaluator.TensorValue(
        core.ShardTensor(
            ctx.result_type,
            evaluator.binary(ctx.op.kind, ctx.args[0].data, ctx.args[1].data),
        )
    )


@verify.register(Binary)
def _(call: "Call", ctx: "verify.Context") -> None:
    lty, rty, dty = (ctx.type_of(arg) for arg in call.args)
    shape = access.broadcast_shapes(lty.shape, rty.shape, raising=False)
    if shape != tuple(dty.shape):
        ctx.error(
            call,
            f"Binary: dst {dty.shape} is not the broadcast of lhs {lty.shape} and rhs {rty.shape}",
        )


@expr.register_op(dialect="T", category="arith")
class Unary(expr.Op):
    """Effect-form pointwise unary operation: ``dst = <kind>(src)``."""

    capability = expr.OpCapability(None)
    execution_mesh = pattern.thread_execution_mesh()

    src = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=IN_RMEM)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=IN_RMEM)
    kind = expr.ParamDef(kind="attribute", annotation=expr.UnaryKind)


@typeinfer.register(Unary)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


access.register_relation(Unary)(access.identity_relations)
