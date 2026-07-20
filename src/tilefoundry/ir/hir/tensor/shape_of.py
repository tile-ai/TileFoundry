from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.expr import Constant
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT


@register_op(name="shape_of")
class ShapeOf(Op):
    x = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(ShapeOf)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    rank_expr = Constant(
        type=TensorType.meta_scalar(), value=len(x_ty.shape)
    )
    return TensorType(
        shape=(rank_expr,), dtype=DType.i64, layout=EMPTY_LAYOUT, storage=None
    )
