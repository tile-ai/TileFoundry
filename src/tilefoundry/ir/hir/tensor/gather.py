from __future__ import annotations

import math

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import Layout
from tilefoundry.ir.types.shard.layout_algebra import prefix_product
from tilefoundry.ir.types.shard.shard_layout import (
    Broadcast,
    Partial,
    ShardLayout,
    Split,
    split_target_axes,
)
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    register_access_relation,
)


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


@register_op
class Gather(Op):
    """Gather along one axis, optionally batched (TF-style ``batch_dims``)."""

    x = ParamDef(kind="input", pattern=Tensor)
    indices = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
    batch_dims = ParamDef(kind="attribute", annotation=int, default=0)


def _norm_axis(axis: int, rank: int, ctx=None, call=None) -> int:
    a = axis + rank if axis < 0 else axis
    if a < 0 or a >= rank:
        msg = f"axis {axis} out of range for rank {rank}"
        if ctx is not None:
            ctx.error(call, msg)
        raise ValueError(f"Gather: {msg}")
    return a


def _check_batch_dims(
    batch_dims: int, axis: int, x_shape: tuple, idx_shape: tuple, ctx, call
) -> int:
    """Validate ``batch_dims`` against the normalized ``axis`` (TF semantics).

    Valid range is ``0 <= batch_dims <= min(axis, rank(index))``; the leading
    ``batch_dims`` dims of ``x`` and ``index`` must match. Returns ``batch_dims``.
    """
    b = batch_dims
    if b < 0:
        ctx.error(call, f"batch_dims {b} must be non-negative")
    if b > axis:
        ctx.error(call, f"batch_dims {b} must be <= axis {axis}")
    if b > len(idx_shape):
        ctx.error(call, f"batch_dims {b} must be <= index rank {len(idx_shape)}")
    if tuple(x_shape[:b]) != tuple(idx_shape[:b]):
        ctx.error(
            call,
            f"batch dims {tuple(x_shape[:b])} of x must match index {tuple(idx_shape[:b])}",
        )
    return b


def _gather_shard_layout(call, ctx, x_ty, axis: int, idx_shape: tuple, out_shape: tuple):
    """Derive a natural contiguous shard layout for a non-batched gather.

    Broadcast and Partial states carry through. A Split on the gathered axis
    becomes ``Partial(sum)``; other Splits are renumbered. Composed or ambiguous
    multi-Split layouts fail closed.
    """
    sl = x_ty.layout
    if not isinstance(sl, ShardLayout):
        return sl
    if not isinstance(sl.layout, Layout):
        ctx.error(
            call,
            f"axis {axis} has a composed shard layout; cannot derive an output layout",
        )
    targets = split_target_axes(sl, tuple(x_ty.shape))
    splits = [(i, t) for i, t in enumerate(targets) if t is not None]
    on_axis = [i for i, t in splits if t == axis]
    if on_axis and len(splits) > 1:
        ctx.error(
            call,
            f"axis {axis} gather over a shard layout with multiple Split "
            f"axes including the gathered axis; cannot derive an output layout",
        )
    natural = Layout(shape=out_shape, strides=prefix_product(out_shape))
    if on_axis:
        mesh_idx = on_axis[0]
        new_attrs = tuple(
            Partial(reduction="sum") if i == mesh_idx else a for i, a in enumerate(sl.attrs)
        )
        return ShardLayout(layout=natural, attrs=new_attrs, mesh=sl.mesh)

    shift = len(idx_shape) - 1
    new_attrs = tuple(
        Split(axis=t + shift if t > axis else t) if isinstance(a, Split) else a
        for a, t in zip(sl.attrs, targets)
    )
    return ShardLayout(layout=natural, attrs=new_attrs, mesh=sl.mesh)


@register_typeinfer(Gather)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    idx_ty = ctx.type_of(call.args[1])
    if idx_ty.dtype not in (DType.i32, DType.i64):
        ctx.error(call, f"index must be an integer tensor (i32/i64), got {idx_ty.dtype}")
    axis = _norm_axis(call.target.axis, len(x_ty.shape), ctx, call)
    b = _check_batch_dims(
        call.target.batch_dims, axis, tuple(x_ty.shape), tuple(idx_ty.shape), ctx, call
    )
    if b > 0 and any(
        isinstance(layout, ShardLayout)
        and not all(isinstance(attr, Broadcast) for attr in layout.attrs)
        for layout in (x_ty.layout, idx_ty.layout)
    ):
        raise NotImplementedError(
            "Gather: batched gather (batch_dims>0) over a sharded operand "
            "(ShardLayout) is not yet supported"
        )

    new_shape = list(x_ty.shape[:axis]) + list(idx_ty.shape[b:]) + list(x_ty.shape[axis + 1 :])
    new_layout = _gather_shard_layout(call, ctx, x_ty, axis, tuple(idx_ty.shape), tuple(new_shape))
    return TensorType(
        shape=tuple(new_shape), dtype=x_ty.dtype, layout=new_layout, storage=x_ty.storage
    )


@register_access_relation(Gather)
def _gather_access_relation(call: "Call", ctx) -> AccessRelations:
    """GLOBAL level: Gather pulls per-index slices.

    GLOBAL level: Gather pulls per-index slices; access pattern is
    data-dependent on the indices arg -> input data is OPAQUE; indices
    identity; output identity.
    """
    idx_rank = len(ctx.type_of(call.args[1]).shape)
    out_rank = len(ctx.type_of(call).shape)
    return AccessRelations(
        inputs=(OPAQUE, _identity(idx_rank)),
        outputs=(_identity(out_rank),),
    )


@register_eval(Gather)
def _eval_gather(ctx):
    x = ctx.args[0].data
    indices = ctx.args[1].data
    axis = _norm_axis(ctx.op.axis, x.dim())
    b = ctx.op.batch_dims
    idx = indices.long()
    if b > 0:
        batch, mid, trail = x.shape[:b], x.shape[b:axis], x.shape[axis + 1 :]
        rem = indices.shape[b:]
        A = x.shape[axis]
        B, M, T, I = math.prod(batch), math.prod(mid), math.prod(trail), math.prod(rem)
        x_r = x.reshape(B, M, A, T)
        idx_exp = idx.reshape(B, 1, I, 1).expand(B, M, I, T)
        out = torch.gather(x_r, 2, idx_exp)
        out = out.reshape(*x.shape[:axis], *rem, *trail)
        return TensorValue(data=out, type=ctx.result_type)
    out = torch.index_select(x, axis, idx.reshape(-1))
    new_shape = x.shape[:axis] + tuple(indices.shape) + x.shape[axis + 1 :]
    return TensorValue(data=out.reshape(new_shape), type=ctx.result_type)
