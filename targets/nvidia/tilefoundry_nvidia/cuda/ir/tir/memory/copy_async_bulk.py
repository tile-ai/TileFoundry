"""Barrier-completing gmem-to-smem bulk asynchronous copy."""

from __future__ import annotations

from tilefoundry.interfaces import access, typeinfer
from tilefoundry.ir import expr, pattern, types


@expr.register_op(dialect="T", category="async", name="copy_async_bulk")
class CopyAsyncBulk(expr.Op):
    """Stage a tile from global to shared memory, completing on a barrier."""

    capability = expr.OpCapability("cp.async.bulk")
    execution_mesh = pattern.thread_execution_mesh()

    src = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.READ, pattern=pattern.tensor_in(expr.StorageKind.GMEM)
    )
    dst = expr.ParamDef(
        kind="input", effect=expr.MemoryEffect.WRITE, pattern=pattern.tensor_in(expr.StorageKind.SMEM)
    )
    barrier = expr.ParamDef(
        kind="input",
        effect=expr.MemoryEffect.READ | expr.MemoryEffect.WRITE,
        pattern=pattern.tensor_in(expr.StorageKind.SMEM),
    )

    between = (pattern.SameConstraint("shape", "src", "dst"), pattern.SameConstraint("dtype", "src", "dst"))


@typeinfer.register(CopyAsyncBulk)
def _(call: "Call", ctx: "TypeInferContext") -> types.UnitType:
    return types.UnitType()


__all__ = ["CopyAsyncBulk"]


@access.register_relation(CopyAsyncBulk)
def bulk_access(call, ctx):
    source = ctx.type_of(call.args[0])
    data = access.identity(len(source.shape))
    barrier = access.state(ctx.type_of(call.args[2]), len(source.shape))
    return access.iterating(source.shape, (data, data, barrier, data))
