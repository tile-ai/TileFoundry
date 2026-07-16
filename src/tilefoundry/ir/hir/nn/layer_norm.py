from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


@register_op(name="layer_norm")
class LayerNorm(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    weight = ParamDef(kind="input", pattern=Tensor)
    bias = ParamDef(kind="input", pattern=Tensor)
    axis = ParamDef(kind="attribute", annotation=int)
    eps = ParamDef(kind="attribute", annotation=float)
@register_typeinfer(LayerNorm)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    weight_ty = ctx.type_of(call.args[1])
    bias_ty = ctx.type_of(call.args[2])
    for arg, ty in (("x", x_ty), ("weight", weight_ty), ("bias", bias_ty)):
        for axis, reduction in enumerate(partial_reductions_by_axis(ty.layout)):
            if reduction is not None:
                ctx.error(
                    call,
                    f"LayerNorm: {arg} carries Partial({reduction}) on mesh "
                    f"axis {axis}, which does not commute; insert reshard({arg}, "
                    "Broadcast) before this consumer",
                )
    return x_ty
