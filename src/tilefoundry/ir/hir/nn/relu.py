from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op
class ReLU(Op):
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(ReLU)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    if any(
        reduction == "sum"
        for reduction in partial_reductions_by_axis(x_ty.layout)
    ):
        ctx.error(
            call,
            "Partial(sum) input on x is unsound (relu is nonlinear, does "
            "not commute with sum) — insert reshard(x, Broadcast) before "
            "this consumer",
        )
    return x_ty
