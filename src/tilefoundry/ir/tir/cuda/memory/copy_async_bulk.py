"""Barrier-completing gmem-to-smem bulk asynchronous copy."""

from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import is_ranked_tensor
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt


@register_op(dialect="T", category="async", name="copy_async_bulk")
class CopyAsyncBulk(Op):
    """Stage a tile from global to shared memory, completing on a barrier."""

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=is_ranked_tensor())
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=is_ranked_tensor())
    barrier = ParamDef(
        kind="input",
        effect=MemoryEffect.READ | MemoryEffect.WRITE,
        pattern=is_ranked_tensor(),
    )


@register_typeinfer(CopyAsyncBulk)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_verify_stmt(CopyAsyncBulk)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    bar = ctx.type_of(call.args[2])
    if src.storage != StorageKind.GMEM:
        ctx.error(call, f"CopyAsyncBulk source must be gmem, got {src.storage}")
    if dst.storage != StorageKind.SMEM:
        ctx.error(call, f"CopyAsyncBulk destination must be smem, got {dst.storage}")
    if bar.storage != StorageKind.SMEM:
        ctx.error(call, f"CopyAsyncBulk barrier must be smem, got {bar.storage}")
    if src.dtype != dst.dtype:
        ctx.error(call, f"CopyAsyncBulk dtype mismatch: {src.dtype} vs {dst.dtype}")
    if src.shape != dst.shape:
        ctx.error(call, f"CopyAsyncBulk shape mismatch: {src.shape} vs {dst.shape}")


__all__ = ["CopyAsyncBulk"]
