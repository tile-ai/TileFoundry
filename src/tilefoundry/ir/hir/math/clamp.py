"""HIR value-form Clamp Op: ``Clamp(x, min_val, max_val)``.

Element-wise clip: ``y = min(max(x, min_val), max_val)``.
"""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.interfaces import access, cost, evaluator, typeinfer
from tilefoundry.ir import expr, pattern, types
from tilefoundry.ir import utils as ir_utils

COMMUTES_WITH = frozenset({"max", "min"})


@expr.register_op(dialect='tf', category='math')
class Clamp(expr.Op):
    """Element-wise clamp: y = min(max(x, min_val), max_val)."""

    x = expr.ParamDef(kind="input", pattern=pattern.is_ranked_tensor())
    min_val = expr.ParamDef(kind="attribute", annotation=float)
    max_val = expr.ParamDef(kind="attribute", annotation=float)


@typeinfer.register(Clamp)
def _(call: "expr.Call", ctx: "TypeInferContext") -> types.TensorType:
    x_ty = ctx.type_of(call.args[0])
    ir_utils.reject_partials("x", x_ty.layout, commutes_with=COMMUTES_WITH)
    return types.TensorType(
        shape=x_ty.shape,
        dtype=x_ty.dtype,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )


@evaluator.register(Clamp)
def eval_clamp(ctx):
    def clamp(x):
        return utils.torch.clamp(x, min=ctx.op.min_val, max=ctx.op.max_val)

    x = ctx.args[0].tensor
    held = clamp(x.tensor) if x.is_global else tuple((p, clamp(t)) for p, t in x.tensor)
    return evaluator.TensorValue(core.ShardTensor(ctx.result_type, held))


__all__ = ["Clamp"]


access.register_relation(Clamp)(access.identity_relations)


@cost.register(Clamp)
def clamp_cost(call: expr.Call, ctx: cost.Context) -> cost.Cost:
    return cost.elementwise(call, ctx)
