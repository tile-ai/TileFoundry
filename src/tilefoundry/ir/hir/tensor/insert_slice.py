"""HIR insert_slice op (dynamic-update-slice)."""

from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue
from tilefoundry.ir.core import Constant, Op, Tuple
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Scalar, Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import require_matching_partial_state
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessMode,
    AccessQuantity,
    AccessRelations,
    BoundaryAccess,
    OperandValue,
    OutputStorage,
    StorageEffectClaim,
    StorageLink,
    WindowAccess,
    elements_of,
    factored_window,
    logical_axes_of,
    moves,
    register_access_relation,
    update_destination,
)
from tilefoundry.visitor_registry.relation_build import identity_access


@register_op(name="insert_slice")
class InsertSlice(Op):
    """Return ``dst`` with ``update`` written into the window at ``offsets``."""

    dst = ParamDef(kind="input", pattern=Tensor)
    update = ParamDef(kind="input", pattern=Tensor)

    offsets = ParamDef(kind="input", pattern=Scalar)


def _insert_slice_storage(call: "Call", ctx) -> StorageEffectClaim | None:
    """The window is written into dst; the result is that same buffer."""
    return update_destination(call, ctx, destination=0)


def _offset_axes(call: "Call", rank: int) -> tuple:
    """Where the window starts on each axis, by value or by reference.

    The offsets arrive either as one tuple naming every axis or as one rank-0
    scalar starting the window on axis zero, and a literal is worth keeping as a
    number: an address that is written down is not a runtime value.
    """
    given = call.args[2]
    if isinstance(given, Tuple):
        elements = given.elements
        return tuple(
            int(item.value)
            if isinstance(item, Constant) and isinstance(item.value, int)
            else OperandValue(operand=2, element=axis)
            for axis, item in enumerate(elements)
        )
    start = (
        int(given.value)
        if isinstance(given, Constant) and isinstance(given.value, int)
        else OperandValue(operand=2)
    )
    return (start, *(0 for _ in range(rank - 1)))


@register_access_relation(InsertSlice)
def _insert_slice_access(call: "Call", ctx) -> AccessRelations:
    """The result is dst with a window replaced, so every index reads itself.

    The window is exactly the update's own shape wherever it lands, so both
    sides answer the same size question, and only the address moves with the
    offsets. Those extents are the ones this participant holds, folded onto the
    logical axes the offsets are stated against. The result states the window
    rather than the container: how big it is and how much of it this occurrence
    wrote are different numbers, and the rest was already there.
    """
    result = ctx.local_type_of(call)
    logical_result = ctx.type_of(call)
    rank = len(logical_result.shape)
    update = ctx.local_type_of(call.args[1])
    logical_update = ctx.type_of(call.args[1])
    held_update = [1] * len(logical_update.shape)
    for position, owner in enumerate(logical_axes_of(update, logical_update)):
        held_update[owner] *= update.shape[position]
    offsets, extents = factored_window(
        _offset_axes(call, rank), tuple(held_update), result, logical_result
    )
    window = elements_of(update)
    kept = elements_of(ctx.local_type_of(call.args[0])) - window
    complement = WindowAccess(offsets, extents, complement=True)
    preserve = StorageLink(
        kind="preserve",
        input=0,
        source=complement,
        output=complement,
        quantity=AccessQuantity(kept, kept),
    )
    return AccessRelations(
        inputs=(
            BoundaryAccess(complement, AccessQuantity(kept, kept), AccessMode.TRANSFER),
            moves(
                WindowAccess(tuple(0 for _ in update.shape), tuple(update.shape)),
                window,
            ),
            *(moves(_scalar_access(ctx, arg), 1) for arg in call.args[2:]),
        ),
        outputs=(
            BoundaryAccess(
                identity_access(len(result.shape)),
                AccessQuantity(window, window),
                AccessMode.WRITE,
                OutputStorage((preserve,)),
            ),
        ),
        storage_effect=_insert_slice_storage(call, ctx),
    )


def _check_axis(ax: int, dst_ext, upd_ext, off_expr, ctx, call) -> None:
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
    if off_ty.dtype not in (DType.i32, DType.i64):
        ctx.error(
            call,
            f"offset for axis {ax} must be an integer scalar, got {off_ty.dtype}",
        )
    d, u = static_dim_value(dst_ext), static_dim_value(upd_ext)
    if d is not None and u is not None and u > d:
        ctx.error(call, f"update extent {u} exceeds dst extent {d} on axis {ax}")
    if isinstance(off_expr, Constant):
        o = int(off_expr.value)
        if o < 0:
            ctx.error(call, f"offset {o} on axis {ax} must be non-negative")
        if d is not None and u is not None and o + u > d:
            ctx.error(
                call,
                f"window [{o}, {o + u}) out of bounds on axis {ax} (dst extent {d})",
            )


@register_typeinfer(InsertSlice)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    dst_ty = ctx.type_of(call.args[0])
    upd_ty = ctx.type_of(call.args[1])
    off_expr = call.args[2]
    rank = len(dst_ty.shape)
    if len(upd_ty.shape) != rank:
        ctx.error(call, f"update rank {len(upd_ty.shape)} must equal dst rank {rank}")
    if dst_ty.dtype != upd_ty.dtype:
        ctx.error(call, f"dst/update dtype mismatch {dst_ty.dtype} vs {upd_ty.dtype}")
    require_matching_partial_state(ctx, call, dst_ty, upd_ty, "dst", "update")
    if isinstance(off_expr, Tuple):
        if len(off_expr.elements) != rank:
            ctx.error(
                call,
                f"offsets tuple length {len(off_expr.elements)} must equal dst rank {rank}",
            )
        for ax, off_el in enumerate(off_expr.elements):
            _check_axis(ax, dst_ty.shape[ax], upd_ty.shape[ax], off_el, ctx, call)
    else:
        off_ty = ctx.type_of(off_expr)
        if len(off_ty.shape) != 0:
            ctx.error(
                call,
                f"offsets must be a rank-0 scalar start or a per-axis tuple, got "
                f"shape {off_ty.shape}",
            )
        if off_ty.dtype not in (DType.i32, DType.i64):
            ctx.error(call, f"offsets must be an integer scalar, got {off_ty.dtype}")
        if rank != 1:
            ctx.error(
                call,
                f"a bare scalar offset applies only to a rank-1 dst; a rank-{rank} "
                "dst needs a per-axis offset tuple",
            )
        _check_axis(0, dst_ty.shape[0], upd_ty.shape[0], off_expr, ctx, call)
    return dst_ty


@register_eval(InsertSlice)
def _eval_insert_slice(ctx):
    dst = ctx.args[0].data
    upd = ctx.args[1].data
    off_val = ctx.args[2]
    if isinstance(off_val, TupleValue):
        starts = [int(e.data.reshape(-1)[0].item()) for e in off_val.elements]
    else:
        starts = [int(off_val.data.reshape(-1)[0].item())]
    sl = []
    for ax, start in enumerate(starts):
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
    return TensorValue(data=out, type=ctx.result_type)


__all__ = ["InsertSlice"]


def _scalar_access(ctx, arg) -> "isl.multi_aff":
    """A scalar operand, at whatever rank its own view gives it.

    A rank-0 value can arrive as a rank-1 position under a layout, and an image
    that names no coordinate cannot be composed with one that has a position.
    """
    held = ctx.local_type_of(arg)
    return identity_access(len(held.shape) if hasattr(held, "shape") else 0)
