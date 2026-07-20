from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer

# Monotone non-decreasing: commutes with max/min, not sum.
_COMMUTES_WITH = frozenset({"max", "min"})


@register_op
class Tanh(Op):
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(Tanh)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    reject_partials(ctx, call, "x", x_ty.layout, commutes_with=_COMMUTES_WITH)
    return x_ty
