"""HIR ``zeros(type)`` callable Op."""

from __future__ import annotations

import torch

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op
class Zeros(Op):
    """Allocate a zero-initialised tensor with the given complete type."""

    type = ParamDef(kind="attribute", annotation=TensorType)


@register_typeinfer(Zeros)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return call.target.type


@register_eval(Zeros)
def _eval_zeros(ctx):

    shape = tuple(resolve_dim(d, ctx.dim_bindings) for d in ctx.op.type.shape)
    data = torch.zeros(shape, dtype=to_torch_dtype(ctx.op.type.dtype), device=ctx.device)
    return TensorValue(data=data, type=ctx.result_type)
