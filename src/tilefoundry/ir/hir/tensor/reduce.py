"""HIR generic Reduce op with kind enum."""

from __future__ import annotations

from math import prod
from types import MappingProxyType

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils

__all__ = ["Reduce"]


@expr.register_op(dialect='tf', category='tensor')
class Reduce(expr.Op):
    """Axis reduction over ``x``, one ``ReduceKind`` per runtime reduce tag."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    axes = expr.ParamDef(kind="attribute", annotation=tuple)
    keepdim = expr.ParamDef(kind="attribute", annotation=bool, default=True)
    kind = expr.ParamDef(kind="attribute", annotation=expr.ReduceKind, default=expr.ReduceKind.MEAN)


def reduced_axes(call: "expr.Call", rank: int) -> tuple:
    return tuple(a % rank if a < 0 else a for a in call.target.axes)


PARTIAL = MappingProxyType(
    {
        expr.ReduceKind.SUM: "sum",
        expr.ReduceKind.MEAN: "sum",
        expr.ReduceKind.MAX: "max",
        expr.ReduceKind.ABS_MAX: "max",
        expr.ReduceKind.MIN: "min",
    }
)
"""What a reduction over a split axis leaves on that mesh axis: each unit's part,
combined by this reduction. A mean's parts are its sums over the whole count."""


@typeinfer.register(Reduce)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    kind = call.target.kind

    commutes_with = frozenset() if kind is expr.ReduceKind.ABS_MAX else frozenset({PARTIAL[kind]})
    ir_utils.reject_partials("x", x_ty.layout, commutes_with=commutes_with)
    keepdim = call.target.keepdim
    rank = len(x_ty.shape)
    reduced = reduced_axes(call, rank)

    new_shape = list(x_ty.shape)
    for a in sorted(reduced, reverse=True):
        if keepdim:
            new_shape[a] = 1
        else:
            new_shape.pop(a)
    out_shape = tuple(new_shape)

    new_layout = types.Layout(out_shape)
    source_shard = types.shard_layout_of(x_ty.layout)
    if source_shard is not None:
        relation = access.relations_of(call, ctx)
        derived = typeinfer.derive_output_shard_layout(
            (x_ty,),
            relation,
            out_shape,
            complete_reduction_dims=frozenset(reduced),
            reduction=PARTIAL[kind],
            fresh_strides=True,
            ctx=ctx,
            storage=x_ty.storage,
        )
        new_layout = (
            derived
            if derived is not None
            else types.ShardLayout.from_logical(
                out_shape,
                source_shard.mesh,
                source_shard.attrs,
                storage=x_ty.storage,
                owners=ctx.target.storage_owners
                if ctx is not None and ctx.target is not None
                else expr.StorageOwners((), {}),
            )
        )

    return types.TensorType(
        shape=out_shape,
        dtype=x_ty.dtype,
        layout=new_layout,
        storage=x_ty.storage,
    )


EMPTY_IDENTITY = (expr.ReduceKind.MAX, expr.ReduceKind.MIN, expr.ReduceKind.ABS_MAX)


def least_representable(dtype: "utils.torch.dtype"):
    """The smallest value *dtype* can hold.

    The smallest value *dtype* can hold: ``False``, ``iinfo.min``, or ``-inf``
    where the dtype round-trips it and ``finfo.min`` where it does not.
    """
    if dtype is utils.torch.bool:
        return False
    if not dtype.is_floating_point:
        return utils.torch.iinfo(dtype).min
    negative_infinity = float("-inf")
    held = utils.torch.tensor(negative_infinity, dtype=utils.torch.float32).to(dtype).to(utils.torch.float32)
    return negative_infinity if held.item() == negative_infinity else utils.torch.finfo(dtype).min


def greatest_representable(dtype: "utils.torch.dtype"):
    """The largest value *dtype* can hold."""
    if dtype is utils.torch.bool:
        return True
    if not dtype.is_floating_point:
        return utils.torch.iinfo(dtype).max
    infinity = float("inf")
    held = utils.torch.tensor(infinity, dtype=utils.torch.float32).to(dtype).to(utils.torch.float32)
    return infinity if held.item() == infinity else utils.torch.finfo(dtype).max


def reduced_count(ctx, axes) -> int:
    """How many elements the reduction covers in the whole input, not in one unit's part.

    A mean over a split axis is Partial sum, so each unit holds its sum over the
    whole count rather than its own mean.
    """
    shape = ctx.args[0].type.shape
    return prod(evaluator.resolve_dim(shape[axis], ctx.dim_bindings) for axis in axes)


@evaluator.register(Reduce)
def eval_reduce(ctx):
    axes = tuple(ctx.op.axes)
    keepdim = ctx.op.keepdim
    kind = ctx.op.kind
    whole = reduced_count(ctx, axes)

    def reduce(x):
        if kind in EMPTY_IDENTITY and any(x.shape[axis] == 0 for axis in axes):
            if kind is expr.ReduceKind.ABS_MAX:
                identity = 0
            elif kind is expr.ReduceKind.MIN:
                identity = greatest_representable(x.dtype)
            else:
                identity = least_representable(x.dtype)
            reduced = list(x.shape)
            for axis in axes:
                reduced[axis] = 1
            out = utils.torch.full(reduced, identity, dtype=x.dtype, device=x.device)
            return out if keepdim else out.squeeze(dim=axes)
        if kind is expr.ReduceKind.MEAN:
            out = x.mean(dim=axes, keepdim=keepdim)
            part = prod(x.shape[axis] for axis in axes)
            return out if part == whole else out * (part / whole)
        if kind is expr.ReduceKind.SUM:
            return x.sum(dim=axes, keepdim=keepdim)
        if kind is expr.ReduceKind.ABS_MAX:
            return x.abs().amax(dim=axes, keepdim=keepdim)
        if kind is expr.ReduceKind.MAX:
            return x.amax(dim=axes, keepdim=keepdim)
        if kind is expr.ReduceKind.MIN:
            return x.amin(dim=axes, keepdim=keepdim)
        raise ValueError(f"evaluator: unsupported ReduceKind {kind}")

    x = ctx.args[0].tensor
    held = reduce(x.tensor) if x.is_global else tuple((p, reduce(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


def kept(shape: tuple) -> int:
    """How many elements a shape of numbers holds."""
    counted = 1
    for extent in shape:
        counted *= extent if isinstance(extent, int) else 1
    return counted


@access.register_relation(Reduce)
def reduce_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
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
    return access.iterating(
        source.shape,
        (access.identity(rank), access.Relation(isl.map(f"{{ [{domain}] -> [{collapses}] }}"))),
    )


@cost.register(Reduce)
def reduce_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    (source,) = cost.input_types(call, ctx)
    output = cost.output_type(call, ctx)
    if not isinstance(source, types.TensorType):
        raise ValueError("Reduce cost requires a tensor input")
    return cost.operation(source.dtype, ir_utils.numel(source), cost.traffic((source,), output))
