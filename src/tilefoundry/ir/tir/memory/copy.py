"""Effect-ful TIR Op ``tir.memory.Copy``.

Copies ``src`` to ``dst``. Memory direction (gmem / smem
/ rmem) is inferred from ``.type.storage`` of each operand. Covers
Load / Store too. The Op is placed in Stmt position as
``Evaluate(Copy, ...)``; the invocation is unit-typed (no result value).
"""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import utils
from tilefoundry.ir.types import LayoutBase, UnitType
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op
class Copy(Op):
    """Copies ``src`` into ``dst`` (in-place memory write)."""

    src = ParamDef(kind="input", effect=MemoryEffect.READ, pattern=utils.operand_tile(0))
    dst = ParamDef(kind="input", effect=MemoryEffect.WRITE, pattern=utils.operand_tile(1))
    rmem_layout = ParamDef(kind="attribute", annotation=LayoutBase, optional=True, default=None)
    smem_layout = ParamDef(kind="attribute", annotation=LayoutBase, optional=True, default=None)

    scope = utils.any_threads()


@register_typeinfer(Copy)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


register_access_relation(Copy)(identity_relations(2))


@register_schedule_eval(Copy)
def _eval_scheduled_copy(ctx):
    return TensorValue(data=ctx.args[0].data, type=ctx.result_type)


@register_verify_stmt(Copy)
def _(call: "Call", ctx: "VerifyContext") -> None:
    src = ctx.type_of(call.args[0])
    dst = ctx.type_of(call.args[1])
    if src.storage == dst.storage and src.shape != dst.shape:
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
