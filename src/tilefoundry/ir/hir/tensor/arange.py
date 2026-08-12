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
from tilefoundry.ir.types.dim import DimSub, ceildiv, is_dim_expr, simplify_dim
from tilefoundry.ir.types.shape_dim import ShapeDim
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    identity_relations,
    register_access_relation,
)


@register_op
class Arange(Op):
    """Generate evenly spaced integer coordinates in ``[start, end)``."""

    end = ParamDef(kind="attribute", annotation=ShapeDim)
    start = ParamDef(kind="attribute", annotation=ShapeDim, default=0)
    step = ParamDef(kind="attribute", annotation=int, default=1)
    dtype = ParamDef(kind="attribute", annotation=DType, default=DType.i64)


register_access_relation(Arange)(identity_relations(0))


@register_typeinfer(Arange)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    op = call.target
    for name, value in (("start", op.start), ("end", op.end)):
        if not is_dim_expr(value):
            ctx.error(call, f"{name} must be a static or symbolic shape dimension")
    if not isinstance(op.step, int) or isinstance(op.step, bool) or op.step <= 0:
        ctx.error(call, f"step must be a positive static integer, got {op.step!r}")
    if op.dtype not in (DType.i32, DType.i64):
        ctx.error(call, f"dtype must be i32 or i64, got {op.dtype}")

    difference = simplify_dim(DimSub, (op.end, op.start))
    static_difference = static_dim_value(difference)
    if static_difference is not None and static_difference < 0:
        ctx.error(
            call,
            f"positive step {op.step} requires end >= start, got "
            f"start={op.start!r}, end={op.end!r}",
        )
    extent = ceildiv(difference, op.step)
    return TensorType(
        shape=(extent,),
        dtype=op.dtype,
        layout=None,
        storage=StorageKind.UMAT,
    )


@register_eval(Arange)
def _eval_arange(ctx):
    start = resolve_dim(ctx.op.start, ctx.dim_bindings)
    end = resolve_dim(ctx.op.end, ctx.dim_bindings)
    data = torch.arange(
        start,
        end,
        ctx.op.step,
        dtype=to_torch_dtype(ctx.op.dtype),
        device=ctx.device,
    )
    return TensorValue(data=data, type=ctx.result_type)


__all__ = ["Arange"]
