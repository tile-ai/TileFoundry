"""HIR pointwise ``Gelu`` Op."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils

COMMUTES_WITH = frozenset()


@expr.register_op(dialect='tf', category='nn')
class Gelu(expr.Op):
    """Gaussian Error Linear Unit.

    Gaussian Error Linear Unit. ``approximate="tanh"`` is the tanh-based
    approximation (HF ``gelu_pytorch_tanh`` / Gemma-2 MLP activation).
    """

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    approximate = expr.ParamDef(kind="attribute", annotation=str, default="tanh")


@typeinfer.register(Gelu)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    ir_utils.reject_partials("x", x_ty.layout, commutes_with=COMMUTES_WITH)
    return x_ty


@evaluator.register(Gelu)
def eval_gelu(ctx):
    def gelu(x):
        return utils.torch.nn.functional.gelu(x, approximate=ctx.op.approximate)

    x = ctx.args[0].tensor
    held = gelu(x.tensor) if x.is_global else tuple((p, gelu(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


access.register_relation(Gelu)(access.identity_relations)


@cost.register(Gelu)
def gelu_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.elementwise(call, ctx)


__all__ = ["Gelu"]
