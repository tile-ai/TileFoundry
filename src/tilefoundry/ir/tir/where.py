"""Effect-form selection of register values under a boolean condition."""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.interfaces import access, typeinfer, verify
from tilefoundry.ir import expr, pattern, types

IN_RMEM = pattern.tensor_in(expr.StorageKind.RMEM)


@expr.register_op(dialect="T", category="arith", name="where")
class Where(expr.Op):
    """Write lhs where cond is true, otherwise rhs, into dst."""

    capability = expr.OpCapability(None)
    execution_mesh = pattern.thread_execution_mesh()

    cond = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.READ, pattern=replace(IN_RMEM, dtype=expr.DType.bool)
    )
    lhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=IN_RMEM)
    rhs = expr.ParamDef(kind="input", effect=expr.MemoryEffect.READ, pattern=IN_RMEM)
    dst = expr.ParamDef(kind="input", effect=expr.MemoryEffect.WRITE, pattern=IN_RMEM)

    between = (pattern.SameConstraint("dtype", "lhs", "rhs"), pattern.SameConstraint("dtype", "lhs", "dst"))


@typeinfer.register(Where)
def _(call, ctx) -> types.UnitType:
    return types.UnitType()


@access.register_relation(Where)
def _(call, ctx):
    shapes = tuple(ctx.type_of(arg).shape for arg in call.args[:3])
    out_shape = access.broadcast_all(shapes)
    output = access.identity(len(out_shape))
    return access.iterating(out_shape, (
        *(access.broadcast(out_shape, shape) for shape in shapes), output, output,
    ))


@verify.register(Where)
def _(call, ctx) -> None:
    cond, lhs, rhs, dst = (ctx.type_of(arg) for arg in call.args)
    if access.broadcast_all((cond.shape, lhs.shape, rhs.shape)) != tuple(dst.shape):
        ctx.error(call, "Where destination must have the broadcast shape of its inputs")


__all__ = ["Where"]
