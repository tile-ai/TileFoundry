from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer


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
        reject_partials(ctx, call, arg, ty.layout)
    return x_ty
