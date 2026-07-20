from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT


@register_op(name="shape_compose")
class ShapeCompose(Op):
    """Assemble a shape from per-axis dims: N rank-0 i64 Exprs → rank-1 shape."""
    is_variadic: ClassVar[bool] = True

    dims = ParamDef(kind="input", pattern=Tensor)
@register_typeinfer(ShapeCompose)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    n = len(call.args)
    return TensorType(
        shape=(i64_const(n),),
        dtype=DType.i64,
        layout=EMPTY_LAYOUT,
        storage=None,
    )
