"""Quantize last-axis groups to FP8 vectors with one f32 scale per group.

The result is ``(x_q, x_scale)``. ``x_q`` preserves the input shape;
``x_scale`` replaces the final extent with ``extent // group``.
"""

from __future__ import annotations

import isl

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import DType, TensorType, TupleType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    register_access_relation,
)


@register_op
class Quant(Op):
    """Per-token-group FP8 quantize. Multi-output (x_q, x_scale)."""

    x = ParamDef(kind="input", pattern=Tensor)
    scheme = ParamDef(kind="attribute", annotation=str, default="per_token_group")
    group = ParamDef(kind="attribute", annotation=int, default=128)
    target_dtype = ParamDef(kind="attribute", annotation=DType, default=DType.fp8e4m3)


@register_typeinfer(Quant)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    x_ty = ctx.type_of(call.args[0])
    if not x_ty.shape:
        ctx.error(call, "x must be at least rank-1")

    reject_partials(ctx, call, "x", x_ty.layout)
    last = x_ty.shape[-1]
    group = call.target.group

    if isinstance(last, int):
        if last % group != 0:
            ctx.error(call, f"last dim {last} not divisible by group={group}")
        scale_last = last // group
    else:
        scale_last = last
    x_q_ty = TensorType(
        shape=x_ty.shape,
        dtype=call.target.target_dtype,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )
    scale_ty = TensorType(
        shape=x_ty.shape[:-1] + (scale_last,),
        dtype=DType.f32,
        layout=x_ty.layout,
        storage=x_ty.storage,
    )
    return TupleType(fields=(x_q_ty, scale_ty))


@register_access_relation(Quant)
def _quant_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL black-box quant.

    - input ``x`` is read element-wise → identity multi_aff over the rank-N
      domain.
    - output ``x_q`` is element-wise identity (same shape).
    - output ``x_scale`` reduces over the in-group offset (last dim divided by
      ``group``); expressed as an isl map ``[..., j] -> [..., j // group]``.
    """
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    group = call.target.group

    dims = ", ".join(f"i{k}" for k in range(rank))
    ident = isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")

    if rank == 0:
        scale_rel = ident  # pragma: no cover
    else:
        outer = ", ".join(f"i{k}" for k in range(rank - 1))
        last = f"i{rank - 1}"
        out_dims = (outer + ", ") if outer else ""
        scale_rel = isl.map(f"{{ [{dims}] -> [{out_dims}floor({last}/{group})] }}")

    return AccessRelations(
        inputs=(ident,),
        outputs=(ident, scale_rel),
    )


__all__ = ["Quant"]
