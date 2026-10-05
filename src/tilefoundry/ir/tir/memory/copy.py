"""Effect-ful TIR Op ``tir.memory.Copy``.

Copies ``src`` to ``dst``. Memory direction (gmem / smem
/ rmem) is inferred from ``.type.storage`` of each operand. Covers
Load / Store too. The Op is placed in Stmt position as
``Evaluate(Copy, ...)``; the invocation is unit-typed (no result value).
"""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_schedule_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op, OpCapability
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.pattern import utils
from tilefoundry.ir.types import LayoutBase, UnitType
from tilefoundry.ir.types.shard_layout import ShardLayout
from tilefoundry.visitor_registry import register_typeinfer, register_verify_stmt
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    BoundaryRelation,
    identity_access,
    identity_relations,
    iterating,
    linearized_view,
    register_access_relation,
)


@register_op
class Copy(Op):
    """Copies ``src`` into ``dst`` (in-place memory write)."""

    capability = OpCapability(None)

    execution_mesh = utils.thread_execution_mesh()

    src = ParamDef(
        kind="input",
        effect=MemoryEffect.READ,
        pattern=utils.operand_tile(0, execution_mesh=execution_mesh),
    )
    dst = ParamDef(
        kind="input",
        effect=MemoryEffect.WRITE,
        pattern=utils.operand_tile(1, execution_mesh=execution_mesh),
    )
    rmem_layout = ParamDef(kind="attribute", annotation=LayoutBase, optional=True, default=None)
    smem_layout = ParamDef(kind="attribute", annotation=LayoutBase, optional=True, default=None)


@register_typeinfer(Copy)
def _(call: "Call", ctx: "TypeInferContext") -> UnitType:
    return UnitType()


@register_access_relation(Copy)
def _copy_access(call: "Call", ctx) -> AccessRelations:
    """Walk ``src``; ``dst`` is reached where the same per-thread buffer holds it.

    Two shapes over one per-thread buffer both regroup row-major onto it
    ([semantic-analysis §3.1](docs/spec/semantic-analysis.md#31-logical-shape-to-layout-domain)),
    so ``dst`` holds each element at the same linear index: a reshape of
    ``src``'s coordinates. Anything else is elementwise.
    """
    src, dst = ctx.type_of(call.args[0]), ctx.type_of(call.args[1])
    if not _is_copyable_shard(src, dst) or src.shape == dst.shape:
        return identity_relations(call, ctx)
    rank = len(src.shape)
    return iterating(
        src.shape,
        AccessRelations(
            inputs=(
                BoundaryRelation(identity_access(rank)),
                BoundaryRelation(linearized_view(tuple(src.shape), tuple(dst.shape))),
            ),
            outputs=(BoundaryRelation(identity_access(rank)),),
        ),
    )


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
