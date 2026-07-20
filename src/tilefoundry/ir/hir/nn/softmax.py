from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op
class SoftMax(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
@register_typeinfer(SoftMax)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    for axis, reduction in enumerate(partial_reductions_by_axis(x_ty.layout)):
        if reduction is not None:
            ctx.error(
                call,
                f"SoftMax: x carries Partial({reduction}) on mesh axis {axis}, "
                "which does not commute; insert reshard(x, Broadcast) before "
                "this consumer",
            )
    return x_ty


@register_eval(SoftMax)
def _eval_softmax(ctx):
    # Reduce in f32 then cast back to the result dtype.
    out = torch.softmax(ctx.args[0].data.float(), dim=ctx.op.axis)
    return TensorValue(
        data=out.to(to_torch_dtype(ctx.result_type.dtype)), type=ctx.result_type
    )
