from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op
class Softplus(Op):
    """Pointwise softplus ``log(1 + e**x)``."""
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(Softplus)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    for axis, reduction in enumerate(partial_reductions_by_axis(x_ty.layout)):
        if reduction == "sum":
            ctx.error(
                call,
                f"Softplus: x carries Partial(sum) on mesh axis {axis}, which "
                "does not commute; insert reshard(x, Broadcast) before this "
                "consumer",
            )
    return x_ty


@register_eval(Softplus)
def _eval_softplus(ctx):

    return TensorValue(
        data=torch.nn.functional.softplus(ctx.args[0].data), type=ctx.result_type
    )
