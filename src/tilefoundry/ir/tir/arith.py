"""Generic TIR effect-form Ops for binary and unary pointwise operations.

Dispatched by kind enum. ``BinaryKind`` and ``UnaryKind`` are owned by
``tilefoundry.ir.core.kinds`` so HIR and TIR carry the same enum values
without remapping.

``Binary`` / ``Unary`` are ``Op`` subclasses (not ``Stmt``); in Stmt
position they are invoked as ``Evaluate(op, args)`` (unit-typed, no
result value), matching the TIR convention shared with
``tir.memory.Copy`` / ``tir.cuda.nn.Mma``.
"""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import BinaryKind, UnaryKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer, register_verify_stmt
from tilefoundry.ir.types import UnitType

__all__ = ["BinaryKind", "Binary", "UnaryKind", "Unary"]

@register_op(dialect="T", category="arith")
class Binary(Op):
    """Effect-form pointwise binary operation: ``dst = lhs <kind> rhs``.

    Spec: tir.md §3.4
    """
    lhs = ParamDef(kind="input", pattern=Tensor)
    rhs = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)
    kind = ParamDef(kind="attribute", annotation=BinaryKind)

@register_typeinfer(Binary)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()

@register_verify_stmt(Binary)
def _(call: "Call", ctx: "VerifyContext") -> None:
    op = call.target
    if not isinstance(op.kind, BinaryKind):
        ctx.error(call, f"Binary: kind must be BinaryKind enum, got {type(op.kind)}")
    lty = ctx.type_of(call.args[0])
    dty = ctx.type_of(call.args[2])
    if lty.shape != dty.shape:
        ctx.error(call, f"Binary shape mismatch: lhs {lty.shape} vs dst {dty.shape}")

@register_op(dialect="T", category="arith")
class Unary(Op):
    """Effect-form pointwise unary operation: ``dst = <kind>(src)``.

    Spec: tir.md §3.4
    """
    src = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)
    kind = ParamDef(kind="attribute", annotation=UnaryKind)

@register_typeinfer(Unary)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()

@register_verify_stmt(Unary)
def _(call: "Call", ctx: "VerifyContext") -> None:
    op = call.target
    if not isinstance(op.kind, UnaryKind):
        ctx.error(call, f"Unary: kind must be UnaryKind enum, got {type(op.kind)}")
    sty = ctx.type_of(call.args[0])
    dty = ctx.type_of(call.args[1])
    if sty.shape != dty.shape:
        ctx.error(call, f"Unary shape mismatch: src {sty.shape} vs dst {dty.shape}")
