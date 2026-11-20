"""HIR ``zeros(type)`` callable Op."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='tensor')
class Zeros(expr.Op):
    """Allocate a zero-initialised tensor with the given complete type."""

    type = expr.ParamDef(kind="attribute", annotation=types.TensorType)


@typeinfer.register(Zeros)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    return call.target.type


@evaluator.register(Zeros)
def eval_zeros(ctx):

    shape = tuple(evaluator.resolve_dim(d, ctx.dim_bindings) for d in ctx.op.type.shape)
    data = utils.torch.zeros(shape, dtype=evaluator.to_torch_dtype(ctx.op.type.dtype), device=ctx.device)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, data))


@access.register_relation(Zeros)
def zeros_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """Nothing is read; every element of the result is written.

    A zero has no source, so there is no input boundary to describe -- which is
    the whole of what this Op does and why it is worth stating rather than
    leaving to a default that would invent one.
    """
    allocated = call.target.type
    return access.iterating(
        allocated.shape,
        (access.identity(len(allocated.shape)),),
    )


@cost.register(Zeros)
def zeros_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """Materialise a tensor of zeros: no arithmetic, one full write."""
    output = cost.output_type(call, ctx)
    return cost.Cost({}, (cost.TrafficBytes(write=ir_utils.tensor_bytes(output)),))


__all__ = ["Zeros"]
