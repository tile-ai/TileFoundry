"""Effect-ful TIR Clamp Op — element-wise per-thread bound."""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt


@register_op(dialect="T", category="arith")
class Clamp(Op):
    """Per-thread element-wise clamp: dst(i) = min(max(src(i), min_val), max_val)."""
    min_val = ParamDef(kind="attribute", annotation=float)
    max_val = ParamDef(kind="attribute", annotation=float)
    src = ParamDef(kind="input", pattern=Tensor)
    dst = ParamDef(kind="input", pattern=Tensor)

@register_typeinfer(Clamp)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()

@register_verify_stmt(Clamp)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src_ty = ctx.type_of(call.args[0])
    dst_ty = ctx.type_of(call.args[1])
    if src_ty.dtype != dst_ty.dtype:
        ctx.error(call, "Clamp: src and dst dtype must match")

__all__ = ["Clamp"]
