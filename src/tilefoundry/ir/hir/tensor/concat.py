"""HIR ``Concat`` Op: join tensors along an axis."""

from __future__ import annotations

from typing import Tuple

import isl

from tilefoundry import core, services, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='tensor')
class Concat(expr.Op):
    """Variadic input op.

    Variadic input op. `Call.args` is a
    plain `tuple[Expr, ...]` of rank-equal TensorType Exprs (NOT a TupleType
    Expr). The lone Param entry documents element type.
    """

    inputs = expr.ParamDef(kind="input", annotation=Tuple[pattern.is_ranked_tensor()], pattern=pattern.is_ranked_tensor())
    axis = expr.ParamDef(kind="attribute", annotation=int)


def sum_dim(a: expr.Expr, b: expr.Expr) -> expr.Expr:

    return services.simplify_dim("add", (a, b))


def normalized_axis(call: "expr.Call", ctx: "TypeInferContext", rank: int) -> int:
    raw_axis = call.target.axis
    axis = raw_axis + rank if raw_axis < 0 else raw_axis
    if not (0 <= axis < rank):
        ctx.error(call, f"axis {raw_axis} out of range for input rank {rank}")
    return axis


@access.register_relation(Concat)
def concat_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """Each input reads its own segment of the result; the result is read whole.

    The segments are the same arithmetic the forward relation states, so the two
    cannot drift: one offset walk, one guard per input.
    """
    type_list = [ctx.type_of(arg) for arg in call.args]
    rank = len(type_list[0].shape)
    axis = normalized_axis(call, ctx, rank)
    extents = tuple(type_.shape[axis] for type_ in type_list)
    if any(not isinstance(extent, int) or isinstance(extent, bool) for extent in extents):
        raise NotImplementedError(
            f"Concat access_relation: concat-axis extents must be static ints, got {extents}"
        )
    dims = [f"d{index}" for index in range(rank)]
    domain_text = ", ".join(dims)
    inputs, offset = [], 0
    for extent in extents:
        reads = list(dims)
        if offset:
            reads[axis] = f"d{axis} - {offset}"
        inputs.append(
            access.Relation(
                isl.map(
                    f"{{ [{domain_text}] -> [{', '.join(reads)}] : "
                    f"{offset} <= d{axis} < {offset + extent} }}"
                )
            )
        )
        offset += extent
    out_shape = (
        *type_list[0].shape[:axis],
        sum(extents),
        *type_list[0].shape[axis + 1 :],
    )
    return access.iterating(
        out_shape,
        (*inputs, access.Relation(isl.multi_aff(f"{{ [{domain_text}] -> [{domain_text}] }}"))),
    )


def reject_concat_axis_splits(call, ctx, type_list, axis: int) -> None:
    for index, type_ in enumerate(type_list):
        layout = types.shard_layout_of(type_.layout)
        if layout is None:
            continue
        targets = types.split_target_axes(layout, type_.shape)
        if any(
            isinstance(attr, types.Split) and targets[mesh_axis] == axis
            for mesh_axis, attr in enumerate(layout.attrs)
        ):
            ctx.error(
                call,
                f"input {index} is Split along concat axis {axis}; use an "
                "explicit Reshard before Concat",
            )


@typeinfer.register(Concat)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    if not call.args:
        ctx.error(call, "Concat requires at least one input")
    type_list = [ctx.type_of(a) for a in call.args]
    base = type_list[0]
    axis = normalized_axis(call, ctx, len(base.shape))
    for index, t in enumerate(type_list[1:], start=1):
        if len(t.shape) != len(base.shape):
            ctx.error(call, f"input {index} rank must match input 0")
        if t.dtype != base.dtype:
            ctx.error(call, f"input {index} dtype must match input 0")
        for dim, (actual, expected) in enumerate(zip(t.shape, base.shape, strict=True)):
            if dim != axis and actual != expected:
                ctx.error(
                    call,
                    f"input {index} shape must match input 0 outside axis {axis}",
                )
    new_shape = list(base.shape)
    for t in type_list[1:]:
        new_shape[axis] = sum_dim(new_shape[axis], t.shape[axis])
    new_shape = services.normalize_dim(tuple(new_shape))

    reject_concat_axis_splits(call, ctx, type_list, axis)
    try:
        relation = access.relations_of(call, ctx)
        layout = typeinfer.derive_output_shard_layout(
            tuple(type_list),
            relation,
            new_shape,
            fresh_strides=True,
            ctx=ctx,
            storage=base.storage,
        )
    except ValueError as error:
        ctx.error(
            call,
            f"cannot derive input ownership: {error}; use an explicit Reshard before Concat",
        )
    if layout is not None:
        ir_utils.require_uniform_partial_slices(type_list, layout, "Concat")
    else:
        layout = types.Layout(new_shape)
    storage = typeinfer.resolve_anchor_storage(ctx, call, *(t.storage for t in type_list))
    return types.TensorType(shape=new_shape, dtype=base.dtype, layout=layout, storage=storage)


@evaluator.register(Concat)
def eval_concat(ctx):
    def concat(*xs):
        return utils.torch.cat(xs, dim=ctx.op.axis)

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        concat(*(x.tensor for x in xs))
        if not placements
        else tuple((p, concat(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@cost.register(Concat)
def concat_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """A concatenation places its inputs side by side and computes nothing.

    Every input element is read once and written once, so the traffic is the
    inputs plus the output -- unlike a slice, which reads only what it keeps.
    """
    return cost.Cost({}, cost.traffic(cost.input_types(call, ctx), cost.output_type(call, ctx)))


__all__ = ["Concat"]
