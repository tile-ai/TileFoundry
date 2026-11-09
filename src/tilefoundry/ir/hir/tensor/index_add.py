"""Pure-value whole-slice indexed accumulation."""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types, utils

from .index_select import norm_dim


@expr.register_op(name="index_add", dialect='tf', category='tensor')
class IndexAdd(expr.Op):
    """Return ``dst`` with ``src`` slices accumulated at ``index`` along ``dim``."""

    dst = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    index = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    src = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    dim = expr.ParamDef(kind="attribute", annotation=int, default=0)


def infer_index_write(
    call: "expr.Call",
    ctx: "TypeInferContext",
    *,
    op_name: str,
    index_dtypes: tuple[expr.DType, ...],
) -> types.TensorType:
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

    dim = norm_dim(call.target.dim, dst_rank, ctx, call)
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
        utils.reject_partials(name, ty.layout)
        reject_splits(ctx, call, op_name, name, ty)
    return dst_ty


def reject_splits(ctx, call: "expr.Call", op_name: str, name: str, type_: types.TensorType) -> None:
    """Refuse a sharded operand here, where the author can still be told why.

    Which rows a participant owns is decided by index *values*, so a split
    destination needs value binding, a payload guard whose coordinates differ
    from the destination's, and per-unit arithmetic that moves with the share --
    three things that only work together. Until they exist, a split is refused
    at the boundary the author wrote rather than costed as if it were whole.
    """
    layout = types.shard_layout_of(type_.layout)
    if layout is None:
        return
    if any(isinstance(attr, types.Split) for attr in layout.attrs):
        ctx.error(
            call,
            f"{op_name}: {name} is Split, and which rows a participant writes "
            "depends on the index values; reshard it to a replicated layout "
            "before the update",
        )


@typeinfer.register(IndexAdd)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    return infer_index_write(
        call,
        ctx,
        op_name="IndexAdd",
        index_dtypes=(expr.DType.i32, expr.DType.i64),
    )


@evaluator.register(IndexAdd)
def eval_index_add(ctx):
    def index_add(dst, index, src):
        return dst.clone().index_add_(norm_dim(ctx.op.dim, dst.dim()), index, src)

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        index_add(*(x.tensor for x in xs))
        if not placements
        else tuple((p, index_add(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["IndexAdd", "infer_index_write"]


@access.register_relation(IndexAdd)
def index_add_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
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
    carried = access.logical_coordinates(dst, logical_dst)
    rows = access.reached_at(rank, dst, logical_dst, carried, free=(dim,))
    named = access.reached_at(rank, index, ctx.type_of(call.args[1]), {}, free=(0,))
    payload = access.reached_at(rank, src, ctx.type_of(call.args[2]), carried, free=(dim,))
    return access.iterating(
        dst.shape,
        (rows, named, payload, rows),
    )


@cost.register(IndexAdd)
def index_add_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    _, index, src = cost.input_types(call, ctx)
    touched = utils.tensor_bytes(src)
    return cost.operation(
        src.dtype,
        utils.numel(src),
        (
            cost.TrafficBytes(read=touched),
            cost.TrafficBytes(read=utils.tensor_bytes(index)),
            cost.TrafficBytes(read=touched),
            cost.TrafficBytes(write=touched),
        ),
    )
