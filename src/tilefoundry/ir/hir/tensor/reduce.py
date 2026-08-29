"""HIR generic Reduce op with kind enum."""

from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import (
    Layout,
    canonical_shard_layout,
    try_c_order_strides,
)
from tilefoundry.ir.types.shard.shard_layout import shard_layout_of
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    identity_access,
    iterating,
    register_access_relation,
    relations_of,
)
from tilefoundry.visitor_registry.shard_propagate import derive_output_shard_layout

__all__ = ["ReduceKind", "Reduce"]


@register_op
class Reduce(Op):
    """Axis reduction over ``x`` (``mean`` / ``sum`` / ``abs_max`` / ``max``)."""

    x = ParamDef(kind="input", pattern=Tensor)
    axes = ParamDef(kind="attribute", annotation=tuple)
    keepdim = ParamDef(kind="attribute", annotation=bool, default=True)
    kind = ParamDef(kind="attribute", annotation=ReduceKind, default=ReduceKind.MEAN)


def _reduced_axes(call: "Call", rank: int) -> tuple:
    return tuple(a % rank if a < 0 else a for a in call.target.axes)


@register_typeinfer(Reduce)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    kind = call.target.kind

    commutes_with = (
        frozenset({"sum"})
        if kind in (ReduceKind.SUM, ReduceKind.MEAN)
        else frozenset({"max"})
        if kind is ReduceKind.MAX
        else frozenset()
    )
    reject_partials(ctx, call, "x", x_ty.layout, commutes_with=commutes_with)
    keepdim = call.target.keepdim
    rank = len(x_ty.shape)
    reduced = _reduced_axes(call, rank)

    new_shape = list(x_ty.shape)
    for a in sorted(reduced, reverse=True):
        if keepdim:
            new_shape[a] = 1
        else:
            new_shape.pop(a)
    out_shape = tuple(new_shape)

    new_layout = (
        None
        if x_ty.layout is None
        else Layout(shape=out_shape, strides=try_c_order_strides(out_shape))
    )
    source_shard = shard_layout_of(x_ty.layout)
    if source_shard is not None:
        relation = relations_of(call, ctx)
        derived = derive_output_shard_layout(
            (x_ty,),
            relation,
            out_shape,
            complete_reduction_dims=frozenset(reduced),
            fresh_strides=True,
        )
        new_layout = (
            derived
            if derived is not None
            else canonical_shard_layout(out_shape, source_shard.mesh, source_shard.attrs)
        )

    return TensorType(
        shape=out_shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


_EMPTY_IDENTITY = (ReduceKind.MAX, ReduceKind.ABS_MAX)


def _least_representable(dtype: "torch.dtype"):
    """The smallest value *dtype* can hold.

    The smallest value *dtype* can hold: ``False``, ``iinfo.min``, or ``-inf``
    where the dtype round-trips it and ``finfo.min`` where it does not.
    """
    if dtype is torch.bool:
        return False
    if not dtype.is_floating_point:
        return torch.iinfo(dtype).min
    negative_infinity = float("-inf")
    held = torch.tensor(negative_infinity, dtype=torch.float32).to(dtype).to(torch.float32)
    return negative_infinity if held.item() == negative_infinity else torch.finfo(dtype).min


@register_eval(Reduce)
def _eval_reduce(ctx):
    x = ctx.args[0].data
    axes = tuple(ctx.op.axes)
    keepdim = ctx.op.keepdim
    kind = ctx.op.kind
    if kind in _EMPTY_IDENTITY and any(x.shape[axis] == 0 for axis in axes):
        identity = 0 if kind is ReduceKind.ABS_MAX else _least_representable(x.dtype)
        reduced = list(x.shape)
        for axis in axes:
            reduced[axis] = 1
        out = torch.full(reduced, identity, dtype=x.dtype, device=x.device)
        return TensorValue(data=out if keepdim else out.squeeze(dim=axes), type=ctx.result_type)
    if kind is ReduceKind.MEAN:
        out = x.mean(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.SUM:
        out = x.sum(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.ABS_MAX:
        out = x.abs().amax(dim=axes, keepdim=keepdim)
    elif kind is ReduceKind.MAX:
        out = x.amax(dim=axes, keepdim=keepdim)
    else:
        raise ValueError(f"evaluator: unsupported ReduceKind {kind}")
    return TensorValue(data=out, type=ctx.result_type)


def _kept(shape: tuple) -> int:
    """How many elements a shape of numbers holds."""
    counted = 1
    for extent in shape:
        counted *= extent if isinstance(extent, int) else 1
    return counted


@register_access_relation(Reduce)
def _reduce_access(call: "Call", ctx) -> AccessRelations:
    """Every source coordinate feeding a result coordinate, read once.

    A reduction walks what it reads, so the source's own positions are the
    coordinates: it reads more of them than it writes, which is the whole of what
    it does, and the collapse happens on the way out. The extents walked are this
    participant's own, a reduced axis being one a layout can split, and the
    correspondence to the result comes from both layouts, since a result
    coordinate names a logical axis and not a position.
    """
    source = ctx.type_of(call.args[0])
    rank = len(source.shape)
    axes = tuple(axis + rank if axis < 0 else axis for axis in call.target.axes)
    out_shape = tuple(
        (1 if axis in axes else extent)
        for axis, extent in enumerate(source.shape)
        if call.target.keepdim or axis not in axes
    )
    carried = {axis: f"d{axis}" for axis in range(rank)}
    surviving = [axis for axis in range(rank) if axis not in axes]
    came_from = (
        {axis: axis for axis in range(rank)} if call.target.keepdim else dict(enumerate(surviving))
    )
    writes_at = [
        "0" if axis not in came_from or came_from[axis] in axes else carried[came_from[axis]]
        for axis in range(len(out_shape))
    ]
    collapses = ", ".join(writes_at)
    domain = ", ".join(f"d{index}" for index in range(rank))
    return iterating(
        source.shape,
        AccessRelations(
            inputs=(BoundaryRelation(identity_access(rank)),),
            outputs=(
                BoundaryRelation(AffineAccess(isl.map(f"{{ [{domain}] -> [{collapses}] }}"))),
            ),
        ),
    )
