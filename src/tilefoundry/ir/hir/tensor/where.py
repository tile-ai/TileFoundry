"""HIR broadcast elementwise selection."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, types
from tilefoundry.ir import utils as ir_utils
from tilefoundry.ir.hir import math


@expr.register_op(dialect='tf', category='tensor')
class Where(expr.Op):
    """Select values from two branches under a boolean condition."""

    condition = expr.ParamDef(kind="input")
    input = expr.ParamDef(kind="input")
    other = expr.ParamDef(kind="input")


@access.register_relation(Where)
def where_access_relation(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    input_types = tuple(ctx.type_of(arg) for arg in call.args)
    shapes = tuple(type_.shape for type_ in input_types)
    out_shape = access.broadcast_all(shapes)
    return access.iterating(out_shape, (
        *(access.broadcast(out_shape, shape) for shape in shapes),
        access.identity(len(out_shape)),
    ))


@typeinfer.register(Where)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    condition, input_, other = (ctx.type_of(arg) for arg in call.args)
    if condition.dtype != expr.DType.bool:
        ctx.error(call, f"condition must have bool dtype, got {condition.dtype}")
    if input_.dtype != other.dtype:
        ctx.error(
            call,
            f"data branch dtype mismatch ({input_.dtype.name} vs {other.dtype.name})",
        )
    for name, type_ in (("condition", condition), ("input", input_), ("other", other)):
        ir_utils.reject_partials(name, type_.layout)

    storage = typeinfer.resolve_anchor_storage(ctx, call, input_.storage, other.storage)
    try:
        relation = access.relations_of(call, ctx)
        out_shape = access.shape_from_relation(
            relation[len(call.args)],
            access.broadcast_all((condition.shape, input_.shape, other.shape)),
        )
        data_shard = typeinfer.derive_output_shard_layout(
            (input_, other), relation[1:], out_shape, ctx=ctx, storage=storage
        )
        layout = (
            data_shard
            if data_shard is not None
            else math.merge_layout(
                types.shard_layout_of(input_.layout) or input_.layout,
                types.shard_layout_of(other.layout) or other.layout,
                out_shape,
            )
        )

        condition_shard = types.shard_layout_of(condition.layout)
        if condition_shard is not None and any(
            isinstance(attr, (types.Split, types.Partial)) for attr in condition_shard.attrs
        ):
            combined = typeinfer.derive_output_shard_layout(
                (condition, input_, other), relation, out_shape, ctx=ctx, storage=storage
            )
            if combined != layout:
                ctx.error(
                    call,
                    "condition sharding does not match the data branches; "
                    "reshard the condition to their distribution",
                )
        elif condition_shard is not None or data_shard is not None:
            layout = typeinfer.derive_output_shard_layout(
                (condition, input_, other), relation, out_shape, ctx=ctx, storage=storage
            )
    except ValueError as error:
        ctx.error(call, f"Where: {error}")

    if layout is None:
        layout = types.Layout(out_shape)
    return types.TensorType(
        shape=out_shape,
        dtype=input_.dtype,
        layout=layout,
        storage=storage,
    )


@evaluator.register(Where)
def eval_where(ctx):
    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        utils.torch.where(*(x.tensor for x in xs))
        if not placements
        else tuple((p, utils.torch.where(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["Where"]


@cost.register(Where)
def where_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """Choosing between two values it already has is a select, not arithmetic."""
    return cost.serviced(call, ctx, "select")
