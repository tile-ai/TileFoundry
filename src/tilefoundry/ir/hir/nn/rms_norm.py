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
from tilefoundry.visitor_registry.access_relation import (
    AccessRelationResult,
    AccessRelations,
    register_access_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.isl_utility import to_domain


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


@register_type_relation(RMSNorm)
def _rms_norm_type_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward relation for ``rms_norm(x, weight, eps)``.

    Modelled on ``SoftMax``'s: the reduction stays fused inside one
    statement (op semantics, not a tiling choice), so the domain is
    ``x``'s batch axes only (``x.shape[:-1]``) and the reduced axis is an
    existential range dim on the read/write maps rather than a domain
    dim. ``weight`` fills that same range; the output map reuses the
    input map's formula (RMSNorm is elementwise-shaped like its input).
    Sharding is the caller's concern (``analysis.poly``'s
    ``_local_type`` narrows a sharded input before it ever reaches here),
    not this relation's.
    """
    x, weight = input_types
    x_shape, w_shape = x.shape, weight.shape
    if len(x_shape) < 1 or len(w_shape) != 1:
        raise NotImplementedError(
            "RMSNorm type_relation: x must be rank >= 1 and weight rank-1, "
            f"got x.shape={x_shape} weight.shape={w_shape}"
        )
    if x_shape[-1] != w_shape[0]:
        raise NotImplementedError(
            f"RMSNorm type_relation: x last dim {x_shape[-1]} != weight "
            f"dim {w_shape[0]}"
        )
    reduce_extent = x_shape[-1]
    if not isinstance(reduce_extent, int) or isinstance(reduce_extent, bool):
        raise NotImplementedError(
            "RMSNorm type_relation: reduction axis must be a static int, "
            f"got {reduce_extent!r} -- a dynamic reduction axis has no isl "
            "representation here"
        )

    batch_shape = x_shape[:-1]
    domain, param_map = to_domain(batch_shape)
    dims = [f"d{i}" for i in range(len(batch_shape))]
    src = "[" + ", ".join(dims) + "]"
    row = ", ".join(dims + ["j"])
    row_map = isl.map(f"{{ {src} -> [{row}] : 0 <= j < {reduce_extent} }}")
    weight_map = isl.map(f"{{ {src} -> [j] : 0 <= j < {reduce_extent} }}")
    return AccessRelationResult(
        domain=domain, maps=(row_map, weight_map, row_map), param_map=param_map
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
