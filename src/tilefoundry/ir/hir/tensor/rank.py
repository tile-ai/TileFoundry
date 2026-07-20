from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op
class Rank(Op):
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(Rank)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return TensorType.meta_scalar()
