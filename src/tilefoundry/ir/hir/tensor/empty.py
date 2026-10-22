"""HIR ``empty(type)`` callable Op."""

from __future__ import annotations

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import TensorType
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelation,
    identity_access,
    iterating,
    register_access_relation,
)
from tilefoundry.visitor_registry.type_defaults import default_type


@register_op
class Empty(Op):
    """Declare tensor storage with undefined contents and the given complete type."""

    type = ParamDef(kind="attribute", annotation=TensorType)
    buffers = ParamDef(kind="attribute", annotation=int, default=1)

    def __init__(self, **attrs) -> None:
        super().__init__(**attrs)
        if type(self.buffers) is not int or self.buffers < 1:
            raise ValueError("empty buffers must be a positive integer")


@register_typeinfer(Empty)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    return default_type(call.target.type, target=ctx.target)


@register_eval(Empty)
def _eval_empty(ctx):

    shape = tuple(resolve_dim(d, ctx.dim_bindings) for d in ctx.op.type.shape)
    data = torch.zeros(shape, dtype=to_torch_dtype(ctx.op.type.dtype), device=ctx.device)
    return TensorValue(data=data, type=ctx.result_type)


@register_access_relation(Empty)
def _empty_access(call: "Call", ctx) -> tuple[AccessRelation, ...]:
    """Declare the result boundary without initializing its contents."""
    return iterating(call.target.type.shape, (identity_access(len(call.target.type.shape)),))
