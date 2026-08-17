"""HIR one-dimensional integer coordinate generation."""

from __future__ import annotations

import torch

from tilefoundry.evaluator.dim import resolve_dim
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import is_dim_expr
from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op
class Arange(Op):
    """Generate a typed one-dimensional sequence from ``start`` and ``step``."""

    type = ParamDef(kind="attribute", annotation=TensorType)
    start = ParamDef(kind="attribute", annotation=ShapeDim, default=0)
    step = ParamDef(kind="attribute", annotation=int, default=1)


register_access_relation(Arange)(identity_relations(0))


@register_typeinfer(Arange)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    op = call.target
    if not isinstance(op.type, TensorType):
        ctx.error(call, "type must be a TensorType")
    if len(op.type.shape) != 1:
        ctx.error(call, f"type must have rank 1, got shape {op.type.shape!r}")
    if not is_dim_expr(op.type.shape[0]) or isinstance(op.type.shape[0], bool):
        ctx.error(call, "type length must be a static or symbolic shape dimension")
    if not is_dim_expr(op.start) or isinstance(op.start, bool):
        ctx.error(call, "start must be a static or symbolic shape dimension")
    if not isinstance(op.step, int) or isinstance(op.step, bool) or op.step <= 0:
        ctx.error(call, f"step must be a positive static integer, got {op.step!r}")
    if op.type.dtype not in (DType.i32, DType.i64):
        ctx.error(call, f"dtype must be i32 or i64, got {op.type.dtype}")
    return op.type


@register_eval(Arange)
def _eval_arange(ctx):
    start = resolve_dim(ctx.op.start, ctx.dim_bindings)
    length = resolve_dim(ctx.op.type.shape[0], ctx.dim_bindings)
    end = start + length * ctx.op.step
    data = torch.arange(
        start,
        end,
        ctx.op.step,
        dtype=to_torch_dtype(ctx.op.type.dtype),
        device=ctx.device,
    )
    return TensorValue(data=data, type=ctx.result_type)


__all__ = ["Arange"]
