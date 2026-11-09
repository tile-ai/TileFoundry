"""HIR ``full_like(x, value)`` callable Op.

Allocate a tensor with the same type (shape / dtype / layout / storage) as ``x``,
filled with a constant scalar ``value``. Shape is taken from ``x`` so a dynamic
(``DimVar``) extent needs no shape literal — this is how the DSL seeds loop-carry
initial values (e.g. ``-inf`` for an online-softmax running max).
"""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(name="full_like", dialect='tf', category='tensor')
class FullLike(expr.Op):
    """Allocate a tensor shaped/typed like ``x``, filled with constant ``value``."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    value = expr.ParamDef(kind="attribute", annotation=float)


@typeinfer.register(FullLike)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    return ctx.type_of(call.args[0])


@evaluator.register(FullLike)
def eval_full_like(ctx):
    def full_like(x):
        return utils.torch.full_like(x, float(ctx.op.value), dtype=evaluator.to_torch_dtype(ctx.result_type.dtype))

    x = ctx.args[0].tensor
    held = full_like(x.tensor) if x.is_global else tuple((p, full_like(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@access.register_relation(FullLike)
def full_like_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """The template's Type is read and none of its elements; every result element is written."""
    shape = tuple(ctx.type_of(call.args[0]).shape)
    rank = len(shape)
    return access.iterating(
        shape,
        (access.unread(rank, rank), access.identity(rank)),
    )


@cost.register(FullLike)
def full_like_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """The template gives its Type and none of its elements; one full write."""
    output = cost.output_type(call, ctx)
    return cost.Cost({}, (cost.TrafficBytes(), cost.TrafficBytes(write=ir_utils.tensor_bytes(output))))


__all__ = ["FullLike"]
