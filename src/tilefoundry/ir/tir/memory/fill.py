"""Effect-ful TIR Op ``tir.memory.Fill``.

Fills ``tensor`` with the rank-0 scalar ``value`` (in-place memory
write). Wrapped by ``Evaluate(Fill, ...)`` in Stmt position.
"""

from __future__ import annotations

from tilefoundry.ir.core import Constant, Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import Scalar, utils
from tilefoundry.ir.types import UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt


@register_op
class Fill(Op):
    """Fills ``tensor`` element-wise with ``value`` (rank-0 scalar)."""

    tensor = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=utils.operand_tile(0))
    value = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=Scalar)
    scope = utils.any_threads()


@register_typeinfer(Fill)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(Fill)
def _(call: "Call", ctx: "VerifyContext") -> None:
    t_ty = ctx.type_of(call.args[0])
    v_ty = ctx.type_of(call.args[1])
    if v_ty.shape != ():
        ctx.error(call, "Fill value must be rank-0 scalar")
    if v_ty.dtype != t_ty.dtype and not (
        isinstance(call.args[1], Constant) and call.args[1].value == 0
    ):
        ctx.error(call, f"Fill dtype mismatch: {v_ty.dtype} vs {t_ty.dtype}")
