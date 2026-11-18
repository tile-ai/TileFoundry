"""HIR pointwise ``ReLU`` Op."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, autodiff, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils

COMMUTES_WITH = frozenset({"max", "min"})


@expr.register_op(dialect='tf', category='nn')
class ReLU(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(ReLU)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    ir_utils.reject_partials("x", x_ty.layout, commutes_with=COMMUTES_WITH)
    return x_ty


@evaluator.register(ReLU)
def eval_relu(ctx):
    x = ctx.args[0].tensor
    held = utils.torch.relu(x.tensor) if x.is_global else tuple((p, utils.torch.relu(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


@autodiff.register(ReLU)
def _(call: expr.Call, cotangent: expr.Expr, ctx: autodiff.Context) -> tuple:
    """``g`` where ``x > 0`` and 0 elsewhere, ``x == 0`` included, as ``torch.relu`` does."""
    (x,) = call.args
    dtype = ctx.type_of(call).dtype
    positive = ctx.call("cast", ctx.call("cmp_gt", x, ctx.constant(0.0, dtype)), dtype=dtype)
    return (ctx.call("mul", cotangent, positive),)


access.register_relation(ReLU)(access.identity_relations)


@cost.register(ReLU)
def relu_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.elementwise(call, ctx)


__all__ = ["ReLU"]
