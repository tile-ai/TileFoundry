"""Dynamic-update-slice: ``insert_slice(dst, update, offsets)``.

Returns ``dst`` with ``update`` written into the window that starts at
``offsets`` (one start per dim) and spans ``update``'s shape — the SSA spelling
of "slice + store", kept distinct from ``scatter`` (data-dependent multi-index).
``update`` has the same rank as ``dst``; ``offsets`` is a rank-length ``i32``
index vector whose entries may be runtime scalars (e.g. a loop induction var).
The value form returns a new ``dst``; an in-place realization is a lowering
concern (the result is anchored on the ``dst`` buffer).

Scope: this milestone implements the **1-D** case (rank-1 ``dst`` / ``update``,
a length-1 ``offsets`` vector). Higher-rank per-dim offsets share this surface
and are rejected at typeinfer until implemented.
"""
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
    # ``offsets`` is the per-dim start vector: one i32 entry per dst axis.
    off_len = _static_len(off_ty.shape)
    if off_len is not None and off_len != len(dst_ty.shape):
        raise TypeError(
            f"insert_slice: offsets length {off_len} must equal dst rank "
            f"{len(dst_ty.shape)}"
        )
    if off_ty.dtype != DType.i32:
        raise TypeError(f"insert_slice: offsets must be i32, got {off_ty.dtype}")
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
