from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op
class Tanh(Op):
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(Tanh)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    for axis, reduction in enumerate(partial_reductions_by_axis(x_ty.layout)):
        if reduction == "sum":
            ctx.error(
                call,
                f"Tanh: x carries Partial(sum) on mesh axis {axis}, which "
                "does not commute; insert reshard(x, Broadcast) before this "
                "consumer",
            )
    return x_ty
