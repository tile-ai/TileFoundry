from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.shard_layout import (
    ShardLayout,
    Split,
    layout_axis_to_tensor_axis,
)


@register_op
class Gather(Op):
    """Gather along one axis, optionally batched (TF-style ``batch_dims``)."""

    x = ParamDef(kind="input", pattern=Tensor)
    indices = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
    batch_dims = ParamDef(kind="attribute", annotation=int, default=0)


def _norm_axis(axis: int, rank: int) -> int:
    a = axis + rank if axis < 0 else axis
    if a < 0 or a >= rank:
        raise TypeError(f"Gather: axis {axis} out of range for rank {rank}")
    return a


def _check_batch_dims(batch_dims: int, axis: int, x_shape: tuple, idx_shape: tuple) -> int:
    """Validate ``batch_dims`` against the normalized ``axis`` (TF semantics).

    Valid range is ``0 <= batch_dims <= min(axis, rank(index))``; the leading
    ``batch_dims`` dims of ``x`` and ``index`` must match. Returns ``batch_dims``.
    """
    b = batch_dims
    if b < 0:
        raise TypeError(f"Gather: batch_dims {b} must be non-negative")
    if b > axis:
        raise TypeError(f"Gather: batch_dims {b} must be <= axis {axis}")
    if b > len(idx_shape):
        raise TypeError(
            f"Gather: batch_dims {b} must be <= index rank {len(idx_shape)}"
        )
    if tuple(x_shape[:b]) != tuple(idx_shape[:b]):
        raise TypeError(
            f"Gather: batch dims {tuple(x_shape[:b])} of x must match index "
            f"{tuple(idx_shape[:b])}"
        )
    return b


def _sliced_shard_layout(x_ty, axis: int, idx_shape: tuple):
    """Return the sliced ``ShardLayout`` for a pure single-index gather.

    A scalar index or rank-1 ``(1,)`` index on a non-sharded axis is a slice:
    scalar indices drop that axis's cute positions, while ``(1,)`` indices keep
    them as size-1 positions. Gather along a Split axis, multi-index gather, or
    composed layout passes the input layout through unchanged.
    """
    sl = x_ty.layout
    if not isinstance(sl, ShardLayout) or not isinstance(sl.layout, Layout):
        return sl
    # Only a scalar index or a rank-1 ``(1,)`` index is a pure slice; any other
    # form (incl. a total-size-1 multi-index like ``(1, 1)``) passes through.
    scalar_idx = idx_shape == ()
    if not scalar_idx and idx_shape != (1,):
        return sl
    cute_shape = sl.layout.shape
    cute_strides = sl.layout.strides
    if cute_strides is None or len(cute_strides) != len(cute_shape):
        return sl
    pos_to_axis = layout_axis_to_tensor_axis(cute_shape, tuple(x_ty.shape))
    if len(pos_to_axis) != len(cute_shape):
        return sl
    sliced = {p for p, a in enumerate(pos_to_axis) if a == axis}
    if not sliced:
        return sl
    # A gather along a ``Split`` (sharded) axis is not a pure slice; pass through.
    if any(isinstance(a, Split) and a.axis in sliced for a in sl.attrs):
        return sl
    if scalar_idx:
        # Drop the sliced positions; remap Split axes onto the survivors.
        keep = [p for p in range(len(cute_shape)) if p not in sliced]
        if not keep:
            return sl
        new_shape = tuple(cute_shape[p] for p in keep)
        new_strides = tuple(cute_strides[p] for p in keep)
        pos_map = {p: i for i, p in enumerate(keep)}
        new_attrs = tuple(
            Split(axis=pos_map[a.axis]) if isinstance(a, Split) else a
            for a in sl.attrs
        )
    else:
        # (1,)-style index: the axis survives at size 1.
        new_shape = tuple(1 if p in sliced else d for p, d in enumerate(cute_shape))
        new_strides = tuple(cute_strides)
        new_attrs = sl.attrs
    return ShardLayout(
        layout=Layout(shape=new_shape, strides=new_strides),
        attrs=new_attrs,
        mesh=sl.mesh,
    )


@register_typeinfer(Gather)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    idx_ty = ctx.type_of(call.args[1])
    if idx_ty.dtype not in (DType.i32, DType.i64):
        raise TypeError(
            f"Gather: index must be an integer tensor (i32/i64), got {idx_ty.dtype}"
        )
    axis = _norm_axis(call.target.axis, len(x_ty.shape))
    b = _check_batch_dims(call.target.batch_dims, axis, tuple(x_ty.shape), tuple(idx_ty.shape))
    if b > 0 and (
        isinstance(x_ty.layout, ShardLayout) or isinstance(idx_ty.layout, ShardLayout)
    ):
        # batch_dims stays a stable interface for a future sharded/collective
        # implementation; the batched path over a sharded operand is not landed.
        raise NotImplementedError(
            "Gather: batched gather (batch_dims>0) over a sharded operand "
            "(ShardLayout) is not yet supported"
        )
    # The gathered axis is replaced by the index's non-batch dims; leading axes
    # and trailing axes of x pass through (batch_dims=0 => full index inserted).
    new_shape = list(x_ty.shape[:axis]) + list(idx_ty.shape[b:]) + list(x_ty.shape[axis + 1:])
    new_layout = _sliced_shard_layout(x_ty, axis, tuple(idx_ty.shape))
    return TensorType(
        shape=tuple(new_shape), dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )


@register_eval(Gather)
def _eval_gather(ctx):
    import math  # noqa: PLC0415

    x = ctx.args[0].data
    indices = ctx.args[1].data
    axis = _norm_axis(ctx.op.axis, x.dim())
    b = ctx.op.batch_dims
    idx = indices.long()
    if b > 0:
        # Batched: out[c.., i.., t..] = x[c.., index[b.., i..], t..]. Flatten to
        # (batch, mid, gathered, trailing) and gather the shared per-batch index
        # across the mid axes, then restore the logical shape.
        batch, mid, trail = x.shape[:b], x.shape[b:axis], x.shape[axis + 1:]
        rem = indices.shape[b:]
        A = x.shape[axis]
        B, M, T, I = math.prod(batch), math.prod(mid), math.prod(trail), math.prod(rem)
        x_r = x.reshape(B, M, A, T)
        idx_exp = idx.reshape(B, 1, I, 1).expand(B, M, I, T)
        out = torch.gather(x_r, 2, idx_exp)
        out = out.reshape(*x.shape[:axis], *rem, *trail)
        return TensorValue(data=out, type=ctx.result_type)
    out = torch.index_select(x, axis, idx.reshape(-1))
    new_shape = x.shape[:axis] + tuple(indices.shape) + x.shape[axis + 1:]
    return TensorValue(data=out.reshape(new_shape), type=ctx.result_type)
