"""HIR ``SoftMax`` Op."""

from __future__ import annotations

import isl

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='nn')
class SoftMax(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    axis = expr.ParamDef(kind="attribute", annotation=int)


@access.register_relation(SoftMax)
def softmax_access(call: "expr.Call", ctx) -> tuple[access.Relation, ...]:
    """One row normalised per iteration, read whole and written whole.

    Every element of a row needs the row's own maximum and sum before any of it
    can be written, so the axis being normalised is not a coordinate this Op is
    asked by: it is free in the images. An identity here would have a reader
    believe each output element depends on one input element, and tile an axis
    that cannot be tiled.
    """
    x_ty = ctx.type_of(call.args[0])
    logical_x = ctx.type_of(call.args[0])
    authored = call.target.axis
    axis = authored + len(logical_x.shape) if authored < 0 else authored
    rows, names, guards = access.normalised_rows(x_ty, logical_x, axis)
    domain = ", ".join(f"d{index}" for index in range(len(rows)))
    where = f" : {' and '.join(guards)}" if guards else ""
    row = access.Relation(isl.map(f"{{ [{domain}] -> [{', '.join(names)}]{where} }}"))
    return access.iterating(
        rows,
        (row, row),
    )


@typeinfer.register(SoftMax)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    ir_utils.reject_partials("x", x_ty.layout)
    return x_ty


@evaluator.register(SoftMax)
def eval_softmax(ctx):
    def softmax(x):
        out = utils.torch.softmax(x.float(), dim=ctx.op.axis)
        return out.to(evaluator.to_torch_dtype(ctx.result_type.dtype))

    x = ctx.args[0].tensor
    held = softmax(x.tensor) if x.is_global else tuple((p, softmax(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@cost.register(SoftMax)
def softmax_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.elementwise(call, ctx)


__all__ = ["SoftMax"]
