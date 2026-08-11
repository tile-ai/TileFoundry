from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.shape_helpers import i64_const
from tilefoundry.ir.types.shard.layout import EMPTY_LAYOUT
from tilefoundry.visitor_registry import register_typeinfer


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


@register_eval(ShapeCompose)
def _eval_shape_compose(ctx):
    data = (
        torch.stack(tuple(arg.data.reshape(()) for arg in ctx.args)).to(torch.int64)
        if ctx.args
        else torch.empty((0,), dtype=torch.int64)
    )
    return TensorValue(data=data, type=ctx.result_type)
