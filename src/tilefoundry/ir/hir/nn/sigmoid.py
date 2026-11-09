"""HIR pointwise ``Sigmoid`` Op."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils

COMMUTES_WITH = frozenset({"max", "min"})


@expr.register_op(dialect='tf', category='nn')
class Sigmoid(expr.Op):
    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(Sigmoid)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    ir_utils.reject_partials("x", x_ty.layout, commutes_with=COMMUTES_WITH)
    return x_ty


@evaluator.register(Sigmoid)
def eval_sigmoid(ctx):
    x = ctx.args[0].tensor
    held = utils.torch.sigmoid(x.tensor) if x.is_global else tuple((p, utils.torch.sigmoid(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


access.register_relation(Sigmoid)(access.identity_relations)


@cost.register(Sigmoid)
def sigmoid_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.elementwise(call, ctx)


__all__ = ["Sigmoid"]
