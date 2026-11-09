"""HIR one-dimensional integer coordinate generation."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import dim, expr, types


@expr.register_op(dialect='tf', category='tensor')
class Arange(expr.Op):
    """Generate a typed one-dimensional sequence from ``start`` and ``step``."""

    type = expr.ParamDef(kind="attribute", annotation=types.TensorType)
    start = expr.ParamDef(kind="attribute", annotation=types.ShapeDim, default=0)
    step = expr.ParamDef(kind="attribute", annotation=int, default=1)


@access.register_relation(Arange)
def arange_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """A sequence made from its own attributes: nothing read, every index written.

    The length is the Op's, not an operand's -- there are no operands -- so the
    space walked comes off the attribute that states the result.
    """
    return access.iterating(
        call.target.type.shape,
        (access.identity(1),),
    )


@typeinfer.register(Arange)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    op = call.target
    if not isinstance(op.type, types.TensorType):
        ctx.error(call, "type must be a TensorType")
    if len(op.type.shape) != 1:
        ctx.error(call, f"type must have rank 1, got shape {op.type.shape!r}")
    if not dim.is_dimension(op.type.shape[0]) or isinstance(op.type.shape[0], bool):
        ctx.error(call, "type length must be a static or symbolic shape dimension")
    if not dim.is_dimension(op.start) or isinstance(op.start, bool):
        ctx.error(call, "start must be a static or symbolic shape dimension")
    if not isinstance(op.step, int) or isinstance(op.step, bool) or op.step <= 0:
        ctx.error(call, f"step must be a positive static integer, got {op.step!r}")
    if op.type.dtype not in (expr.DType.i32, expr.DType.i64):
        ctx.error(call, f"dtype must be i32 or i64, got {op.type.dtype}")
    return op.type


@evaluator.register(Arange)
def eval_arange(ctx):
    start = evaluator.resolve_dim(ctx.op.start, ctx.dim_bindings)
    length = evaluator.resolve_dim(ctx.op.type.shape[0], ctx.dim_bindings)
    end = start + length * ctx.op.step
    data = utils.torch.arange(
        start,
        end,
        ctx.op.step,
        dtype=evaluator.to_torch_dtype(ctx.op.type.dtype),
        device=ctx.device,
    )
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, data))


__all__ = ["Arange"]


@cost.register(Arange)
def arange_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    """Coordinates are synthesized metadata until a consumer materializes them."""
    return cost.Cost({}, cost.idle(call))
