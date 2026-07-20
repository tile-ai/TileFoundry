"""Effect-ful TIR Op ``tir.memory.Copy``.

Copies ``source`` to ``destination``. Memory direction (gmem / smem
/ rmem) is inferred from ``.type.storage`` of each operand. Covers
Load / Store too. The Op is placed in Stmt position as
``Evaluate(Copy, ...)``; the invocation is unit-typed (no result value).
"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import UnitType
from tilefoundry.ir.types.shard.shard_layout import ShardLayout
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt


@register_op
class Copy(Op):
    """Copies ``source`` into ``destination`` (in-place memory write)."""
    source = ParamDef(kind="input", pattern=Tensor)
    destination = ParamDef(kind="input", pattern=Tensor)

@register_typeinfer(Copy)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()

@register_verify_stmt(Copy)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if src.storage == dst.storage and src.shape != dst.shape:
        # Allow shape mismatch when both sides carry the same ShardLayout
        # (reshape / broadcast↔split — logical shape changes, per-thread
        # buffer is the same copyable extent).
        if not _is_copyable_shard(src, dst):
            ctx.error(call, f"Copy shape mismatch: {src.shape} vs {dst.shape}")
    if src.dtype != dst.dtype:
        ctx.error(call, f"Copy dtype mismatch: {src.dtype} vs {dst.dtype}")

def _is_copyable_shard(src_ty, dst_ty) -> bool:
    """Both sides carry a ShardLayout describing the same per-thread buffer."""
    src_sl = getattr(src_ty, "layout", None)
    dst_sl = getattr(dst_ty, "layout", None)
    if not (isinstance(src_sl, ShardLayout) and isinstance(dst_sl, ShardLayout)):
        return False
    return src_sl.layout == dst_sl.layout
