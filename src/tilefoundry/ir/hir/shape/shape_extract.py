from __future__ import annotations

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="shape_extract")
class ShapeExtract(Op):
    """Extract one axis from a shape value."""

    shape = ParamDef(kind="input", pattern=Tensor)
    index = ParamDef(kind="attribute", annotation=int)


@register_typeinfer(ShapeExtract)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return TensorType.meta_scalar()


@register_eval(ShapeExtract)
def _eval_shape_extract(ctx):
    return TensorValue(data=ctx.args[0].data[ctx.op.index], type=ctx.result_type)
