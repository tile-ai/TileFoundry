"""Barrier-completing gmem-to-smem bulk asynchronous copy."""

from __future__ import annotations

from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import SameConstraint, utils
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_access,
    iterating,
    register_access_relation,
    state_access,
)


@register_op(dialect="T", category="async", name="copy_async_bulk")
class CopyAsyncBulk(Op):
    """Stage a tile from global to shared memory, completing on a barrier."""

    capability = OpCapability("cp.async.bulk")
    execution_mesh = utils.thread_execution_mesh()

    src = ParamDef(
        kind="input", effect=MemoryEffect.READ, pattern=utils.tensor_in(StorageKind.GMEM)
    )
    dst = ParamDef(
        kind="input", effect=MemoryEffect.WRITE, pattern=utils.tensor_in(StorageKind.SMEM)
    )
    barrier = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=utils.tensor_in(StorageKind.SMEM),
    )

    between = (SameConstraint("shape", "src", "dst"), SameConstraint("dtype", "src", "dst"))


@register_typeinfer(CopyAsyncBulk)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


__all__ = ["CopyAsyncBulk"]


@register_access_relation(CopyAsyncBulk)
def _bulk_access(call, ctx):
    source = ctx.type_of(call.args[0])
    data = identity_access(len(source.shape))
    barrier = state_access(ctx.type_of(call.args[2]), len(source.shape))
    return iterating(source.shape, (data, data, barrier, data))
