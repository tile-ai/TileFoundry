"""Define fused last-axis RMS normalization.

The rank-one weight matches the final input extent; preceding dimensions pass
through, including symbolic expressions. Computation accumulates in f32 and
casts back to the input dtype, while the weight may use another dtype.
"""

from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import reject_partials
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    AccessRelations,
    AffineAccess,
    BoundaryRelation,
    factored_image,
    iterating,
    logical_term,
    normalised_rows,
    register_access_relation,
)


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return AffineAccess(isl.map("{ [] -> [] }"))
    dims = ", ".join(f"i{i}" for i in range(rank))
    return AffineAccess(isl.map(f"{{ [{dims}] -> [{dims}] }}"))


@register_op(name="rms_norm")
class RMSNorm(Op):
    x = ParamDef(kind="input", pattern=Tensor)
    weight = ParamDef(kind="input", pattern=Tensor)
    eps = ParamDef(kind="attribute", annotation=float, default=1e-6)


@register_typeinfer(RMSNorm)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    x_ty = ctx.type_of(call.args[0])
    w_ty = ctx.type_of(call.args[1])

    if len(x_ty.shape) < 1:
        ctx.error(call, f"x must be rank ≥ 1, got shape {x_ty.shape}")
    if len(w_ty.shape) != 1:
        ctx.error(call, f"weight must be rank-1, got shape {w_ty.shape}")
    if x_ty.shape[-1] != w_ty.shape[0]:
        ctx.error(call, f"x last dim {x_ty.shape[-1]} != weight dim {w_ty.shape[0]}")

    for arg, ty in (("x", x_ty), ("weight", w_ty)):
        reject_partials(ctx, call, arg, ty.layout)

    return x_ty


@register_access_relation(RMSNorm)
def _rms_norm_relation(call: "Call", ctx) -> AccessRelations:
    """GLOBAL level: one row normalised per iteration, read whole.

    A normalisation is asked once per row, because every element of a row needs
    the whole row's sum before any of it can be written. So the axis it
    normalises is not a coordinate this Op is asked by: it is free in the images,
    and the row is what each boundary reaches. The weight matches that axis and
    nothing else, so every row reaches all of it -- one weight element each.
    """
    x_ty = ctx.type_of(call.args[0])
    w_ty = ctx.type_of(call.args[1])
    logical_x = ctx.type_of(call.args[0])
    normalised = len(logical_x.shape) - 1
    rows, names, guards = normalised_rows(x_ty, logical_x, normalised)
    domain = ", ".join(f"d{index}" for index in range(len(rows)))
    where = f" : {' and '.join(guards)}" if guards else ""
    element = f"{{ [{domain}] -> [{', '.join(names)}]{where} }}"
    across = ", ".join(
        factored_image(
            [logical_term(names, x_ty, logical_x, normalised)],
            w_ty,
            ctx.type_of(call.args[1]),
        )
    )
    return iterating(
        rows,
        AccessRelations(
            inputs=(
                BoundaryRelation(AffineAccess(isl.map(element))),
                BoundaryRelation(AffineAccess(isl.map(f"{{ [{domain}] -> [{across}]{where} }}"))),
            ),
            outputs=(BoundaryRelation(AffineAccess(isl.map(element))),),
        ),
    )


@register_eval(RMSNorm)
def _eval_rms_norm(ctx):

    xf = ctx.args[0].data.float()
    wf = ctx.args[1].data.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(ms + ctx.op.eps) * wf
    return TensorValue(data=out.to(to_torch_dtype(ctx.result_type.dtype)), type=ctx.result_type)
