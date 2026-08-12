from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.expr import Constant
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="shape_of")
class ShapeOf(Op):
    x = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(ShapeOf)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    rank_expr = Constant(type=TensorType.umat_scalar(), value=len(x_ty.shape))
    return TensorType.umat_tensor((rank_expr,), DType.i64)


@register_eval(ShapeOf)
def _eval_shape_of(ctx):
    data = torch.tensor(tuple(ctx.args[0].data.shape), dtype=torch.int64)
    return TensorValue(data=data, type=ctx.result_type)
