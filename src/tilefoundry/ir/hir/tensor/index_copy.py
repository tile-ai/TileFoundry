"""Pure-value whole-slice indexed copy."""

from __future__ import annotations

from tilefoundry import core
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types, utils

from .index_add import infer_index_write
from .index_select import norm_dim


@expr.register_op(name="index_copy", dialect='tf', category='tensor')
class IndexCopy(expr.Op):
    """Return ``dst`` with ``src`` slices copied to ``index`` along ``dim``."""

    dst = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    index = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    src = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    dim = expr.ParamDef(kind="attribute", annotation=int, default=0)


@typeinfer.register(IndexCopy)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    return infer_index_write(
        call,
        ctx,
        op_name="IndexCopy",
        index_dtypes=(expr.DType.i64,),
    )


@evaluator.register(IndexCopy)
def eval_index_copy(ctx):
    def index_copy(dst, index, src):
        return dst.clone().index_copy_(norm_dim(ctx.op.dim, dst.dim()), index, src)

    xs = tuple(arg.tensor for arg in ctx.args)
    placements = evaluator.common_placements(*xs)
    held = (
        index_copy(*(x.tensor for x in xs))
        if not placements
        else tuple((p, index_copy(*(x.to_local(p) for x in xs))) for p in placements)
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["IndexCopy"]


@access.register_relation(IndexCopy)
def index_copy_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """The rows the index names are replaced; the container around them is kept.

    Two questions, and the index answers only one. Which rows are reached its
    values decide, so those boundaries cover every row the axis could legally
    name; no relation here holds the deciding element. Where the container lives
    it answers not at all: every coordinate of the destination is reached, being
    either a row this replaces or one it keeps. The payload's own row is `i`
    where the destination's is `index[i]`, so from here it too is a row nobody
    named, and reaching all of them is the payload.
    """
    dst = ctx.type_of(call.args[0])
    index = ctx.type_of(call.args[1])
    src = ctx.type_of(call.args[2])
    logical_dst = ctx.type_of(call.args[0])
    rank = len(dst.shape)
    dim = call.target.dim + rank if call.target.dim < 0 else call.target.dim
    identity = access.identity(rank)
    carried = access.logical_coordinates(dst, logical_dst)
    rows = access.reached_at(rank, dst, logical_dst, carried, free=(dim,))
    named = access.reached_at(rank, index, ctx.type_of(call.args[1]), {}, free=(0,))
    payload = access.reached_at(rank, src, ctx.type_of(call.args[2]), carried, free=(dim,))
    return access.iterating(
        dst.shape,
        (identity, named, payload, rows),
    )


@cost.register(IndexCopy)
def index_copy_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    _, index, src = cost.input_types(call, ctx)
    touched = utils.tensor_bytes(src)
    return cost.Cost(
        {},
        (
            cost.TrafficBytes(),
            cost.TrafficBytes(read=utils.tensor_bytes(index)),
            cost.TrafficBytes(read=touched),
            cost.TrafficBytes(write=touched),
        ),
    )
