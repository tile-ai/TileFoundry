from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import partial_reductions


@register_op
class Sigmoid(Op):
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(Sigmoid)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if "sum" in partial_reductions(x_ty.layout):
        ctx.error(
            call,
            "Partial(sum) input on x is unsound (sigmoid is nonlinear, does "
            "not commute with sum) — insert reshard(x, Broadcast) before "
            "this consumer",
        )
    return x_ty


@register_eval(Sigmoid)
def _eval_sigmoid(ctx):

    return TensorValue(data=torch.sigmoid(ctx.args[0].data), type=ctx.result_type)
