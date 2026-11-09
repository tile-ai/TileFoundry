"""Fused SiLU/Swish HIR Op."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils


@expr.register_op(dialect='tf', category='nn')
class Silu(expr.Op):
    """Pointwise ``x * sigmoid(x)``, as one fused op."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())


@typeinfer.register(Silu)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])

    ir_utils.reject_partials("x", x_ty.layout)
    return x_ty


@evaluator.register(Silu)
def eval_silu(ctx):
    silu = utils.torch.nn.functional.silu
    x = ctx.args[0].tensor
    held = silu(x.tensor) if x.is_global else tuple((p, silu(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["Silu"]


access.register_relation(Silu)(access.identity_relations)


@cost.register(Silu)
def silu_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.elementwise(call, ctx)
