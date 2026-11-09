"""HIR ``empty(type)`` callable Op."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, types


@expr.register_op(dialect='tf', category='tensor')
class Empty(expr.Op):
    """Declare tensor storage with undefined contents and the given complete type."""

    type = expr.ParamDef(kind="attribute", annotation=types.TensorType)
    buffers = expr.ParamDef(kind="attribute", annotation=int, default=1)

    def __init__(self, **attrs) -> None:
        super().__init__(**attrs)
        if type(self.buffers) is not int or self.buffers < 1:
            raise ValueError("empty buffers must be a positive integer")


@typeinfer.register(Empty)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    return call.target.type


@evaluator.register(Empty)
def eval_empty(ctx):

    shape = tuple(evaluator.resolve_dim(d, ctx.dim_bindings) for d in ctx.op.type.shape)
    data = utils.torch.zeros(shape, dtype=evaluator.to_torch_dtype(ctx.op.type.dtype), device=ctx.device)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, data))


@access.register_relation(Empty)
def empty_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """Declare the result boundary without initializing its contents."""
    return access.iterating(call.target.type.shape, (access.identity(len(call.target.type.shape)),))


@cost.register(Empty)
def empty_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """Storage declarations perform no arithmetic or memory traffic."""
    return cost.Cost({}, cost.idle(call))


__all__ = ["Empty"]
