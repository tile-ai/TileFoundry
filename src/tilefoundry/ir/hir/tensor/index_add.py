"""Pure-value whole-slice indexed accumulation."""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.hir.tensor.index_select import _norm_dim
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard import shard_layout_of
from tilefoundry.ir.types.shard.shard_layout import Split
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    BoundaryRelation,
    iterating,
    logical_coordinates,
    reached_at,
    register_access_relation,
)


@register_op(name="index_add")
class IndexAdd(Op):
    """Return ``dst`` with ``src`` slices accumulated at ``index`` along ``dim``."""

    dst = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="input", pattern=Tensor)
    src = ParamDef(kind="input", pattern=Tensor)
    dim = ParamDef(kind="attribute", annotation=int, default=0)


def _infer_index_write(
    call: "Call",
    ctx: "TypeInferContext",
    *,
    op_name: str,
    index_dtypes: tuple[DType, ...],
) -> TensorType:
    dst_ty = ctx.type_of(call.args[0])
    index_ty = ctx.type_of(call.args[1])
    src_ty = ctx.type_of(call.args[2])

    if len(index_ty.shape) != 1:
        ctx.error(call, f"{op_name}: index must be 1-D, got shape {index_ty.shape}")
    if index_ty.dtype not in index_dtypes:
        allowed = " or ".join(dtype.name for dtype in index_dtypes)
        ctx.error(call, f"{op_name}: index must have dtype {allowed}, got {index_ty.dtype}")

    dst_rank = len(dst_ty.shape)
    if len(src_ty.shape) != dst_rank:
        ctx.error(
            call,
            f"{op_name}: src rank {len(src_ty.shape)} must equal dst rank {dst_rank}",
        )
    if src_ty.dtype != dst_ty.dtype:
        ctx.error(
            call,
            f"{op_name}: dst/src dtype mismatch {dst_ty.dtype} vs {src_ty.dtype}",
        )

    dim = _norm_dim(call.target.dim, dst_rank, ctx, call)
    for axis, (dst_extent, src_extent) in enumerate(zip(dst_ty.shape, src_ty.shape)):
        if axis != dim and dst_extent != src_extent:
            ctx.error(
                call,
                f"{op_name}: src shape {src_ty.shape} must match dst shape "
                f"{dst_ty.shape} outside dim {dim}; mismatch at dim {axis}",
            )
    if index_ty.shape[0] != src_ty.shape[dim]:
        ctx.error(
            call,
            f"{op_name}: index length {index_ty.shape[0]} must equal "
            f"src.shape[{dim}] {src_ty.shape[dim]}",
        )

    for name, ty in (("dst", dst_ty), ("index", index_ty), ("src", src_ty)):
        reject_partials(ctx, call, name, ty.layout)
        _reject_splits(ctx, call, op_name, name, ty)
    return dst_ty


def _reject_splits(ctx, call: "Call", op_name: str, name: str, type_: TensorType) -> None:
    """Refuse a sharded operand here, where the author can still be told why.

    Which rows a participant owns is decided by index *values*, so a split
    destination needs value binding, a payload guard whose coordinates differ
    from the destination's, and per-unit arithmetic that moves with the share --
    three things that only work together. Until they exist, a split is refused
    at the boundary the author wrote rather than costed as if it were whole.
    """
    layout = shard_layout_of(type_.layout)
    if layout is None:
        return
    if any(isinstance(attr, Split) for attr in layout.attrs):
        ctx.error(
            call,
            f"{op_name}: {name} is Split, and which rows a participant writes "
            "depends on the index values; reshard it to a replicated layout "
            "before the update",
        )


@register_typeinfer(IndexAdd)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return _infer_index_write(
        call,
        ctx,
        op_name="IndexAdd",
        index_dtypes=(DType.i32, DType.i64),
    )


@register_eval(IndexAdd)
def _eval_index_add(ctx):
    dst, index, src = (arg.data for arg in ctx.args)
    dim = _norm_dim(ctx.op.dim, dst.dim())
    return TensorValue(
        data=dst.clone().index_add_(dim, index, src),
        type=ctx.result_type,
    )


__all__ = ["IndexAdd"]


@register_access_relation(IndexAdd)
def _index_add_access(call: "Call", ctx) -> AccessRelations:
    """The rows the index names are read, added to, and written back.

    Two questions, and the index answers only one. Which rows are reached its
    values decide, so those boundaries cover every row the axis could legally
    name; no relation here holds the deciding element. Where the container lives
    it answers not at all: every coordinate of the destination is reached, being
    either a row this replaces or one it keeps. The payload's own row is `i`
    where the destination's is `index[i]`, so from here it too is a row nobody
    named, and reaching all of them is the payload.
    """
    dst = ctx.type_of(call.args[0])
    index = ctx.type_of(call.args[1])
    src = ctx.type_of(call.args[2])
    logical_dst = ctx.type_of(call.args[0])
    rank = len(dst.shape)
    dim = call.target.dim + rank if call.target.dim < 0 else call.target.dim
    carried = logical_coordinates(dst, logical_dst)
    rows = reached_at(rank, dst, logical_dst, carried, free=(dim,))
    named = reached_at(rank, index, ctx.type_of(call.args[1]), {}, free=(0,))
    payload = reached_at(rank, src, ctx.type_of(call.args[2]), carried, free=(dim,))
    return iterating(
        dst.shape,
    AccessRelations(
            inputs=(
                BoundaryRelation(rows),
                BoundaryRelation(named),
                BoundaryRelation(payload),
            ),
            outputs=(
                BoundaryRelation(rows),
            ),
        ),
    )
