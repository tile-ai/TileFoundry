"""Generic TIR effect-form Ops for binary and unary pointwise operations.

Dispatched by kind enum. ``BinaryKind`` and ``UnaryKind`` are owned by
``tilefoundry.ir.core.kinds`` so HIR and TIR carry the same enum values
without remapping.

``Binary`` / ``Unary`` are ``Op`` subclasses (not ``Stmt``); in Stmt
position they are invoked as ``Evaluate(op, args)`` (unit-typed, no
result value), matching the TIR convention shared with
``tir.memory.Copy``.
"""

from __future__ import annotations

from tilefoundry.evaluator.kernels import binary
from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import utils
from tilefoundry.ir.types import StorageKind, UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    broadcast_shapes,
    identity_relations,
    register_access_relation,
)

__all__ = ["BinaryKind", "Binary", "UnaryKind", "Unary"]

_IN_RMEM = utils.tensor_in(StorageKind.RMEM)


@register_op(dialect="T", category="arith")
class Binary(Op):
    """Effect-form pointwise binary operation: ``dst = lhs <kind> rhs``."""

    capability = OpCapability(None)
    execution_mesh = utils.thread_execution_mesh()

    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_IN_RMEM)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_IN_RMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_IN_RMEM)
    kind = ParamDef(kind="attribute", annotation=BinaryKind)


@register_typeinfer(Binary)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


register_access_relation(Binary)(identity_relations)


@register_schedule_eval(Binary)
def _eval_scheduled_binary(ctx):
    return TensorValue(
        data=binary(ctx.op.kind, ctx.args[0].data, ctx.args[1].data), type=ctx.result_type
    )


@register_verify_stmt(Binary)
def _(call: "Call", ctx: "VerifyContext") -> None:
    lty, rty, dty = (ctx.type_of(arg) for arg in call.args)
    shape = broadcast_shapes(lty.shape, rty.shape, raising=False)
    if shape != tuple(dty.shape):
        ctx.error(
            call,
            f"Binary: dst {dty.shape} is not the broadcast of lhs {lty.shape} and rhs {rty.shape}",
        )


@register_op(dialect="T", category="arith")
class Unary(Op):
    """Effect-form pointwise unary operation: ``dst = <kind>(src)``."""

    capability = OpCapability(None)
    execution_mesh = utils.thread_execution_mesh()

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_IN_RMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_IN_RMEM)
    kind = ParamDef(kind="attribute", annotation=UnaryKind)


@register_typeinfer(Unary)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


register_access_relation(Unary)(identity_relations)
