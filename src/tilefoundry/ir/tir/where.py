"""Effect-form selection of register values under a boolean condition."""

from __future__ import annotations

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir.tensor.where import _broadcast_all, _maps
from tilefoundry.ir.pattern import utils
from tilefoundry.ir.types import DType, StorageKind, UnitType
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import iterating, register_access_relation

_IN_RMEM = utils.tensor_in(StorageKind.RMEM)


@register_op(dialect="T", category="arith", name="where")
class Where(Op):
    """Write lhs where cond is true, otherwise rhs, into dst."""

    capability = OpCapability(None)
    execution_mesh = utils.thread_execution_mesh()

    cond = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_IN_RMEM)
    lhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_IN_RMEM)
    rhs = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=_IN_RMEM)
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=_IN_RMEM)


@register_typeinfer(Where)
def _(call, ctx) -> UnitType:
    return UnitType()


@register_access_relation(Where)
def _(call, ctx):
    shapes = tuple(ctx.type_of(arg).shape for arg in call.args[:3])
    boundaries = _maps(shapes)
    return iterating(_broadcast_all(shapes), (*boundaries, boundaries[-1]))


@register_verify_stmt(Where)
def _(call, ctx) -> None:
    cond, lhs, rhs, dst = (ctx.type_of(arg) for arg in call.args)
    if cond.dtype != DType.bool:
        ctx.error(call, "Where condition must have bool dtype")
    if lhs.dtype != rhs.dtype or lhs.dtype != dst.dtype:
        ctx.error(call, "Where data branches and destination must have matching dtypes")
    if _broadcast_all((cond.shape, lhs.shape, rhs.shape)) != tuple(dst.shape):
        ctx.error(call, "Where destination must have the broadcast shape of its inputs")


__all__ = ["Where"]
