"""HIR ``full_like(x, value)`` callable Op.

Allocate a tensor with the same type (shape / dtype / layout / storage) as ``x``,
filled with a constant scalar ``value``. Shape is taken from ``x`` so a dynamic
(``DimVar``) extent needs no shape literal — this is how the DSL seeds loop-carry
initial values (e.g. ``-inf`` for an online-softmax running max).
"""
from __future__ import annotations

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer


@register_op(name="full_like")
class FullLike(Op):
    """Allocate a tensor shaped/typed like ``x``, filled with constant ``value``."""
    x = ParamDef(kind="input", pattern=Tensor)
    value = ParamDef(kind="attribute", annotation=float)


@register_typeinfer(FullLike)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return ctx.type_of(call.args[0])


@register_eval(FullLike)
def _eval_full_like(ctx):
    x = ctx.args[0].data
    data = torch.full_like(x, float(ctx.op.value), dtype=to_torch_dtype(ctx.result_type.dtype))
    return TensorValue(data=data, type=ctx.result_type)
