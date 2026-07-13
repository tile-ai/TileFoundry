"""HIR insert_slice op (dynamic-update-slice)."""
from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue
from tilefoundry.ir.core import Constant, Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Scalar, Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.hir.tensor.tuple import Tuple
from tilefoundry.ir.types import DType, TensorType


@register_op(name="insert_slice")
class InsertSlice(Op):
    """Dynamic-update-slice: return ``dst`` with ``update`` written into the window at ``offsets``."""
    dst = ParamDef(kind="input", pattern=Tensor)
    update = ParamDef(kind="input", pattern=Tensor)
    # A rank-0 scalar start (rank-1 dst) or a tuple of per-axis rank-0 scalars
    # (rank-N); the parser lifts the tuple literal to an ``hir.tensor.Tuple``.
    offsets = ParamDef(kind="input", pattern=Scalar)


def _static(dim) -> "int | None":
    """A shape dim as a plain ``int`` if statically known, else ``None``."""
    v = getattr(dim, "value", dim)
    return v if isinstance(v, int) else None


def _check_axis(ax: int, dst_ext, upd_ext, off_expr, ctx) -> None:
    """Per-axis window checks: the update extent must fit, and a *literal*
    (``Constant``) offset must place an in-bounds, non-negative window. A
    runtime offset is deferred to the eval bounds guard."""
    off_ty = ctx.type_of(off_expr)
    if off_ty.shape != ():
        raise TypeError(
            f"insert_slice: offset for axis {ax} must be a rank-0 scalar, got "
            f"shape {off_ty.shape}"
        )
    if off_ty.dtype not in (DType.i32, DType.i64):
        raise TypeError(
            f"insert_slice: offset for axis {ax} must be an integer scalar, got "
            f"{off_ty.dtype}"
        )
    d, u = _static(dst_ext), _static(upd_ext)
    if d is not None and u is not None and u > d:
        raise TypeError(
            f"insert_slice: update extent {u} exceeds dst extent {d} on axis {ax}"
        )
    if isinstance(off_expr, Constant):
        o = int(off_expr.value)
        if o < 0:
            raise TypeError(
                f"insert_slice: offset {o} on axis {ax} must be non-negative"
            )
        if d is not None and u is not None and o + u > d:
            raise TypeError(
                f"insert_slice: window [{o}, {o + u}) out of bounds on axis {ax} "
                f"(dst extent {d})"
            )


@register_typeinfer(InsertSlice)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    dst_ty = ctx.type_of(call.args[0])
    upd_ty = ctx.type_of(call.args[1])
    off_expr = call.args[2]
    rank = len(dst_ty.shape)
    if len(upd_ty.shape) != rank:
        raise TypeError(
            f"insert_slice: update rank {len(upd_ty.shape)} must equal dst rank {rank}"
        )
    if dst_ty.dtype != upd_ty.dtype:
        raise TypeError(
            f"insert_slice: dst/update dtype mismatch {dst_ty.dtype} vs {upd_ty.dtype}"
        )
    if isinstance(off_expr, Tuple):
        # rank-N: one rank-0 scalar offset per axis.
        if len(off_expr.elements) != rank:
            raise TypeError(
                f"insert_slice: offsets tuple length {len(off_expr.elements)} must "
                f"equal dst rank {rank}"
            )
        for ax, off_el in enumerate(off_expr.elements):
            _check_axis(ax, dst_ty.shape[ax], upd_ty.shape[ax], off_el, ctx)
    else:
        # 1-D compatibility: a bare rank-0 scalar start applies only to rank-1.
        off_ty = ctx.type_of(off_expr)
        if len(off_ty.shape) != 0:
            raise TypeError(
                f"insert_slice: offsets must be a rank-0 scalar start or a per-axis "
                f"tuple, got shape {off_ty.shape}"
            )
        if off_ty.dtype not in (DType.i32, DType.i64):
            raise TypeError(
                f"insert_slice: offsets must be an integer scalar, got {off_ty.dtype}"
            )
        if rank != 1:
            raise TypeError(
                f"insert_slice: a bare scalar offset applies only to a rank-1 dst; a "
                f"rank-{rank} dst needs a per-axis offset tuple"
            )
        _check_axis(0, dst_ty.shape[0], upd_ty.shape[0], off_expr, ctx)
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
