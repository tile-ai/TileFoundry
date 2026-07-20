"""Fused RMSNorm HIR Op — Qwen3 extension op.

Semantics: ``x * rsqrt(mean(x**2, axis=-1, keepdim=True) + eps) * weight``.
Compute in f32, cast back to input dtype.

Typeinfer is rank-agnostic: ``x`` may have any rank ≥ 1 and
``weight`` must be rank-1 with the same length as ``x.shape[-1]``;
all batch dimensions ``x.shape[:-1]`` (including symbolic ``DimVar``
/ dim-arithmetic ``DimExpr`` entries) flow through verbatim. The
``weight`` dtype is permitted to differ from ``x.dtype`` (typical
Qwen / LLaMA-family pattern: ``bf16`` activations with ``f32`` scale
vector); the op semantics keep the f32 internal accumulate and cast
back to ``x.dtype`` on output.
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
from tilefoundry.visitor_registry.access_relation import AccessRelations, register_access_relation


def _identity(rank: int) -> "isl.multi_aff":
    if rank == 0:
        return isl.multi_aff("{ [] -> [] }")
    dims = ", ".join(f"i{i}" for i in range(rank))
    return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")


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
    # rms_norm normalizes across an axis (non-monotonic); no reduction commutes.
    for arg, ty in (("x", x_ty), ("weight", w_ty)):
        reject_partials(ctx, call, arg, ty.layout)

    # Output preserves x's full shape (batch dims flow verbatim,
    # including DimVar / dim-arithmetic entries) and x's dtype. The
    # weight may carry a different dtype (typically f32 scale on a
    # bf16 input); the internal f32 accumulate cast-back is op
    # semantics, not a type constraint.
    return x_ty


@register_access_relation(RMSNorm)
def _rms_norm_relation(call: "Call", ctx) -> AccessRelations:
    """GLOBAL level: x identity, weight identity (broadcast along last dim
    treated as identity at GLOBAL black-box; reduction is internal to the
    op)."""
    x_ty = ctx.type_of(call.args[0])
    rank = len(x_ty.shape)
    return AccessRelations(
        inputs=(_identity(rank), _identity(1)),
        outputs=(_identity(rank),),
    )


@register_eval(RMSNorm)
def _eval_rms_norm(ctx):


    xf = ctx.args[0].data.float()
    wf = ctx.args[1].data.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(ms + ctx.op.eps) * wf
    return TensorValue(
        data=out.to(to_torch_dtype(ctx.result_type.dtype)), type=ctx.result_type
    )
