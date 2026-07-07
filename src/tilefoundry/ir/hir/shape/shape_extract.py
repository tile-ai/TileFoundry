from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT


@register_op(name="shape_extract")
class ShapeExtract(Op):
    """Extract one axis from a shape value."""
    shape = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="attribute", annotation=int)
@register_typeinfer(ShapeExtract)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return TensorType(shape=(), dtype=DType.i64, layout=EMPTY_LAYOUT, storage=None)
