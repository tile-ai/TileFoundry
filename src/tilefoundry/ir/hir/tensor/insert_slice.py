"""HIR insert_slice op (dynamic-update-slice)."""

from __future__ import annotations

import isl

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
    OPAQUE,
    AccessRelations,
    StorageClaim,
    register_access_relation,
    update_destination,
)
from tilefoundry.visitor_registry.relation_build import identity_map


@register_op(name="insert_slice")
class InsertSlice(Op):
    """Return ``dst`` with ``update`` written into the window at ``offsets``."""

    dst = ParamDef(kind="input", pattern=Tensor)
    update = ParamDef(kind="input", pattern=Tensor)

    offsets = ParamDef(kind="input", pattern=Scalar)


def _insert_slice_storage(call: "Call", ctx) -> StorageClaim | None:
    """The window is written into dst; the result is that same buffer."""
    return update_destination(call, ctx, destination=0)


def _literal_offsets(call: "Call") -> tuple[int, ...] | None:
    """The window's per-axis offsets, when every one of them is written down.

    An offset that arrives as a runtime value has no number to build a map from,
    and this carrier cannot bind one to an isl parameter yet.
    """
    found = []
    for offset in call.args[2:]:
        if not isinstance(offset, Constant) or not isinstance(offset.value, int):
            return None
        found.append(int(offset.value))
    return tuple(found)


@register_access_relation(InsertSlice)
def _insert_slice_access(call: "Call", ctx) -> AccessRelations:
    """The result is dst with a window replaced, so every index reads itself.

    Where the window sits decides which of the two the index came from. Written
    down, that is one affine guard per axis on each side; arriving at run time,
    it is the one thing here that cannot be said exactly, and only the two
    boundaries it controls become opaque.
    """
    rank = len(ctx.type_of(call).shape)
    identity = identity_map(rank)
    offsets = _literal_offsets(call)
    if offsets is None:
        return AccessRelations(
            inputs=(OPAQUE, OPAQUE, *(identity_map(0) for _ in call.args[2:])),
            outputs=(identity,),
            storage=_insert_slice_storage(call, ctx),
        )
    sizes = tuple(ctx.type_of(call.args[1]).shape)
    dims = [f"d{index}" for index in range(rank)]
    domain = ", ".join(dims)
    inside = " and ".join(
        f"{start} <= d{axis} < {start + size}"
        for axis, (start, size) in enumerate(zip(offsets, sizes))
    )
    window = ", ".join(f"d{axis} - {start}" for axis, start in enumerate(offsets))
    return AccessRelations(
        inputs=(
            isl.map(f"{{ [{domain}] -> [{domain}] : not ({inside}) }}"),
            isl.map(f"{{ [{domain}] -> [{window}] : {inside} }}"),
            *(identity_map(0) for _ in call.args[2:]),
        ),
        outputs=(identity,),
        storage=_insert_slice_storage(call, ctx),
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
