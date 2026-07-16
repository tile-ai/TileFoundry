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

import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.visitor_registry.shard_propagate import partial_reductions_by_axis


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
        raise TypeError(
            f"RMSNorm: x must be rank ≥ 1, got shape {x_ty.shape}"
        )
    if len(w_ty.shape) != 1:
        raise TypeError(
            f"RMSNorm: weight must be rank-1, got shape {w_ty.shape}"
        )
    if x_ty.shape[-1] != w_ty.shape[0]:
        raise TypeError(
            f"RMSNorm: x last dim {x_ty.shape[-1]} != weight dim {w_ty.shape[0]}"
        )
    if any(
        reduction is not None
        for reduction in partial_reductions_by_axis(x_ty.layout)
    ):
        raise TypeError(
            "RMSNorm: Partial input on x is unsound (rms_norm normalizes "
            "across an axis, a non-monotonic combination that does not "
            "commute with any reduction) — insert reshard(x, Broadcast) "
            "before this consumer"
        )

    # Output preserves x's full shape (batch dims flow verbatim,
    # including DimVar / dim-arithmetic entries) and x's dtype. The
    # weight may carry a different dtype (typically f32 scale on a
    # bf16 input); the internal f32 accumulate cast-back is op
    # semantics, not a type constraint.
    return x_ty


@register_eval(RMSNorm)
def _eval_rms_norm(ctx):


    xf = ctx.args[0].data.float()
    wf = ctx.args[1].data.float()
    ms = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(ms + ctx.op.eps) * wf
    return TensorValue(
        data=out.to(to_torch_dtype(ctx.result_type.dtype)), type=ctx.result_type
    )
