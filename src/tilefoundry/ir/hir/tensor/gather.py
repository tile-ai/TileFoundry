from __future__ import annotations

import math

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
    Partial,
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


def _prefix_product(shape: tuple) -> tuple:
    """Exclusive prefix product of ``shape`` (shard.md §3 default stride)."""
    strides = []
    acc = 1
    for d in shape:
        strides.append(acc)
        acc *= int(d)
    return tuple(strides)


def _sliced_shard_layout(call, ctx, x_ty, axis: int, idx_shape: tuple):
    """Derive the output ``ShardLayout`` for a non-batched gather.

    ``strides=None`` derives as a fresh contiguous layout first. A ``Split``
    bound to the gathered axis becomes ``Partial(sum)`` (each device holds a
    zero-filled partial of the gathered rows). Otherwise a scalar or ``(1,)``
    index derives the sliced layout. Anything left over fails closed via
    ``ctx.error``.
    """
    sl = x_ty.layout
    if not isinstance(sl, ShardLayout):
        return sl
    if not isinstance(sl.layout, Layout):
        ctx.error(
            call,
            f"axis {axis} has a composed shard layout; cannot derive an "
            f"output layout",
        )
    cute_shape = sl.layout.shape
    cute_strides = sl.layout.strides
    if cute_strides is None:
        cute_strides = _prefix_product(cute_shape)
    elif len(cute_strides) != len(cute_shape):
        ctx.error(
            call,
            f"axis {axis} shard layout strides {cute_strides!r} do not "
            f"match cute shape {cute_shape!r}",
        )
    pos_to_axis = layout_axis_to_tensor_axis(cute_shape, tuple(x_ty.shape))
    if len(pos_to_axis) != len(cute_shape):
        ctx.error(
            call, f"axis {axis} shard layout positions do not resolve to tensor axes"
        )
    sliced = {p for p, a in enumerate(pos_to_axis) if a == axis}
    if not sliced:
        ctx.error(call, f"axis {axis} not found in the shard layout")

    splits = [a for a in sl.attrs if isinstance(a, Split)]
    splits_on_axis = [a for a in splits if a.axis in sliced]

    if len(splits) == 1 and splits_on_axis:
        # Masked-gather: every device already holds the true value at the
        # rows it owns and a zero row elsewhere, so summing the per-device
        # partials across the mesh axis reconstructs the true gather.
        mesh_idx = next(
            i for i, a in enumerate(sl.attrs) if isinstance(a, Split) and a.axis in sliced
        )
        new_attrs = tuple(
            Partial(reduction="sum") if i == mesh_idx else a
            for i, a in enumerate(sl.attrs)
        )
        first, last = min(sliced), max(sliced)
        new_shape = cute_shape[:first] + tuple(idx_shape) + cute_shape[last + 1:]
        return ShardLayout(
            layout=Layout(shape=new_shape, strides=_prefix_product(new_shape)),
            attrs=new_attrs,
            mesh=sl.mesh,
        )

    if len(splits) > 1:
        ctx.error(
            call,
            f"axis {axis} gather over a shard layout with multiple Split "
            f"axes {tuple(s.axis for s in splits)}; cannot derive an output layout",
        )
    scalar_idx = idx_shape == ()
    if idx_shape != () and idx_shape != (1,):
        if splits:
            ctx.error(
                call,
                f"axis {axis} gather with index shape {idx_shape} combined "
                f"with Split({splits[0].axis}) on a different axis; cannot "
                f"derive an output layout",
            )
        ctx.error(
            call,
            f"axis {axis} gather with index shape {idx_shape} has no "
            f"derivable shard-layout rule",
        )
    if scalar_idx:
        # Drop the sliced positions; remap Split axes onto the survivors.
        keep = [p for p in range(len(cute_shape)) if p not in sliced]
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
    new_layout = _sliced_shard_layout(call, ctx, x_ty, axis, tuple(idx_ty.shape))
    return TensorType(
        shape=tuple(new_shape), dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )


@register_eval(Gather)
def _eval_gather(ctx):
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
