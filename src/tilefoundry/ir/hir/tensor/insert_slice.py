"""HIR insert_slice op (dynamic-update-slice)."""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types, utils


@expr.register_op(name="insert_slice", dialect='tf', category='tensor')
class InsertSlice(expr.Op):
    """Return ``dst`` with ``update`` written into the window at ``offsets``."""

    dst = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    update = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())

    offsets = expr.ParamDef(kind="input", pattern=pattern.is_scalar_tensor())


def offset_axes(call: "expr.Call", rank: int) -> tuple:
    """Where the window starts on each axis, as a number or as the value it is.

    The offsets arrive either as one tuple naming every axis or as one rank-0
    scalar starting the window on axis zero. A literal is worth keeping as a
    number -- an address written down is not a runtime value -- and anything else
    is kept as the operand element it is, so a relation can name it.
    """
    given = call.args[2]
    if isinstance(given, expr.Tuple):
        return tuple(
            int(item.value) if isinstance(item, expr.Constant) and isinstance(item.value, int) else item
            for item in given.elements
        )
    start = (
        int(given.value) if isinstance(given, expr.Constant) and isinstance(given.value, int) else given
    )
    return (start, *(0 for _ in range(rank - 1)))


@access.register_relation(InsertSlice)
def insert_slice_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """The result is dst with a window replaced, so every index reads itself.

    The window is exactly the update's own shape wherever it lands, so both
    sides answer the same size question, and only the address moves with the
    offsets. Those extents are the ones this participant holds, folded onto the
    logical axes the offsets are stated against. The result states the window
    rather than the container: how big it is and how much of it this occurrence
    wrote are different numbers, and the rest was already there.
    """
    result = ctx.type_of(call.args[0])
    rank = len(result.shape)
    update = ctx.type_of(call.args[1])
    offsets = offset_axes(call, rank)
    complement, written = access.placed_window(
        offsets, tuple(update.shape), rank, within=tuple(result.shape)
    )
    read_update = access.window_source(
        offsets,
        rank,
        update,
        update,
        access.logical_coordinates(result, result),
        extents=tuple(update.shape),
    )
    return access.iterating(
        result.shape,
        (
            complement,
            read_update,
            *(access.control_read(rank, ctx, arg) for arg in call.args[2:]),
            written,
        ),
    )


def check_axis(ax: int, dst_ext, upd_ext, off_expr, ctx, call) -> None:
    """Per-axis window checks: the update extent must fit.

    Per-axis window checks: the update extent must fit, and a *literal*
    (``Constant``) offset must place an in-bounds, non-negative window. A
    runtime offset is deferred to the eval bounds guard.
    """
    off_ty = ctx.type_of(off_expr)
    if off_ty.shape != ():
        ctx.error(
            call,
            f"offset for axis {ax} must be a rank-0 scalar, got shape {off_ty.shape}",
        )
    if off_ty.dtype not in (expr.DType.i32, expr.DType.i64):
        ctx.error(
            call,
            f"offset for axis {ax} must be an integer scalar, got {off_ty.dtype}",
        )
    d, u = expr.static_dim_value(dst_ext), expr.static_dim_value(upd_ext)
    if d is not None and u is not None and u > d:
        ctx.error(call, f"update extent {u} exceeds dst extent {d} on axis {ax}")
    if isinstance(off_expr, expr.Constant):
        o = int(off_expr.value)
        if o < 0:
            ctx.error(call, f"offset {o} on axis {ax} must be non-negative")
        if d is not None and u is not None and o + u > d:
            ctx.error(
                call,
                f"window [{o}, {o + u}) out of bounds on axis {ax} (dst extent {d})",
            )


@typeinfer.register(InsertSlice)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    dst_ty = ctx.type_of(call.args[0])
    upd_ty = ctx.type_of(call.args[1])
    off_expr = call.args[2]
    rank = len(dst_ty.shape)
    if len(upd_ty.shape) != rank:
        ctx.error(call, f"update rank {len(upd_ty.shape)} must equal dst rank {rank}")
    if dst_ty.dtype != upd_ty.dtype:
        ctx.error(call, f"dst/update dtype mismatch {dst_ty.dtype} vs {upd_ty.dtype}")
    utils.require_matching_partial_state(dst_ty, upd_ty, "dst", "update")
    if isinstance(off_expr, expr.Tuple):
        if len(off_expr.elements) != rank:
            ctx.error(
                call,
                f"offsets tuple length {len(off_expr.elements)} must equal dst rank {rank}",
            )
        for ax, off_el in enumerate(off_expr.elements):
            check_axis(ax, dst_ty.shape[ax], upd_ty.shape[ax], off_el, ctx, call)
    else:
        off_ty = ctx.type_of(off_expr)
        if len(off_ty.shape) != 0:
            ctx.error(
                call,
                f"offsets must be a rank-0 scalar start or a per-axis tuple, got "
                f"shape {off_ty.shape}",
            )
        if off_ty.dtype not in (expr.DType.i32, expr.DType.i64):
            ctx.error(call, f"offsets must be an integer scalar, got {off_ty.dtype}")
        if rank != 1:
            ctx.error(
                call,
                f"a bare scalar offset applies only to a rank-1 dst; a rank-{rank} "
                "dst needs a per-axis offset tuple",
            )
        check_axis(0, dst_ty.shape[0], upd_ty.shape[0], off_expr, ctx, call)
    return dst_ty


def inserted_slice(dst, upd, *starts):
    sl = []
    for ax, start in enumerate(int(start.reshape(-1)[0].item()) for start in starts):
        n = upd.shape[ax]
        if start < 0 or start + n > dst.shape[ax]:
            raise ValueError(
                f"insert_slice: window [{start}, {start + n}) out of bounds on axis "
                f"{ax} for dst extent {dst.shape[ax]}"
            )
        sl.append(slice(start, start + n))
    win = tuple(sl)
    out = dst.clone()
    out[win] = upd.reshape(out[win].shape).to(out.dtype)
    return out


@evaluator.register(InsertSlice)
def eval_insert_slice(ctx):
    offsets = ctx.args[2]
    offsets = offsets.elements if isinstance(offsets, evaluator.TupleValue) else (offsets,)
    xs = tuple(arg.tensor for arg in (ctx.args[0], ctx.args[1], *offsets))
    placements = evaluator.common_placements(*xs)
    held = (
        inserted_slice(*(x.tensor for x in xs))
        if not placements
        else tuple((p, inserted_slice(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["InsertSlice"]


@cost.register(InsertSlice)
def insert_slice_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """Read and write the update window without charging the untouched dst."""
    _, update, offsets = cost.input_types(call, ctx)
    window = utils.tensor_bytes(update)
    return cost.Cost(
        {},
        (
            cost.TrafficBytes(),
            cost.TrafficBytes(read=window),
            cost.TrafficBytes(read=utils.tensor_bytes(offsets)),
            cost.TrafficBytes(write=window),
        ),
    )
