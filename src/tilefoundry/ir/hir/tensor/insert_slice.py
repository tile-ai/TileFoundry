"""HIR insert_slice op (dynamic-update-slice)."""
from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType


@register_op(name="insert_slice")
class InsertSlice(Op):
    """Dynamic-update-slice: return ``dst`` with ``update`` written into the window at ``offsets``."""
    dst = ParamDef(kind="input", pattern=Tensor)
    update = ParamDef(kind="input", pattern=Tensor)
    offsets = ParamDef(kind="input", pattern=Tensor)


def _static_len(shape) -> "int | None":
    """The single static extent of a rank-1 shape, or ``None`` if dynamic."""
    if len(shape) != 1:
        return None
    d = shape[0]
    v = getattr(d, "value", d)
    return v if isinstance(v, int) else None


@register_typeinfer(InsertSlice)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    dst_ty = ctx.type_of(call.args[0])
    upd_ty = ctx.type_of(call.args[1])
    off_ty = ctx.type_of(call.args[2])
    if len(upd_ty.shape) != len(dst_ty.shape):
        raise TypeError(
            f"insert_slice: update rank {len(upd_ty.shape)} must equal dst rank "
            f"{len(dst_ty.shape)}"
        )
    if len(dst_ty.shape) != 1:
        raise NotImplementedError(
            "insert_slice: only the 1-D case is supported currently; N-D per-dim "
            "offsets are planned on the same surface"
        )
    if dst_ty.dtype != upd_ty.dtype:
        raise TypeError(
            f"insert_slice: dst/update dtype mismatch {dst_ty.dtype} vs {upd_ty.dtype}"
        )
    # The 1-D case takes a single scalar start: a rank-0 integer tensor for a
    # runtime value, or a compile-time integer literal. A rank>=1 offset (incl.
    # ``(1,)`` / ``(1, 1)``) is not the surface.
    if len(off_ty.shape) != 0:
        raise TypeError(
            f"insert_slice: offsets must be a rank-0 scalar start for the 1-D "
            f"case, got shape {off_ty.shape}"
        )
    if off_ty.dtype not in (DType.i32, DType.i64):
        raise TypeError(
            f"insert_slice: offsets must be an integer scalar, got {off_ty.dtype}"
        )
    # Static in-bounds check when both the update extent and (constant) offset
    # are known; a dynamic offset is checked at runtime by the caller.
    dst_n, upd_n = _static_len(dst_ty.shape), _static_len(upd_ty.shape)
    if dst_n is not None and upd_n is not None and upd_n > dst_n:
        raise TypeError(
            f"insert_slice: update extent {upd_n} exceeds dst extent {dst_n}"
        )
    return dst_ty


@register_eval(InsertSlice)
def _eval_insert_slice(ctx):
    dst = ctx.args[0].data
    upd = ctx.args[1].data
    offs = ctx.args[2].data.reshape(-1)
    start = int(offs[0].item())
    n = upd.shape[0]
    if start < 0 or start + n > dst.shape[0]:
        raise ValueError(
            f"insert_slice: window [{start}, {start + n}) out of bounds for dst "
            f"extent {dst.shape[0]}"
        )
    out = dst.clone()
    out[start:start + n] = upd.reshape(out[start:start + n].shape).to(out.dtype)
    return TensorValue(data=out, type=ctx.result_type)


__all__ = ["InsertSlice"]
