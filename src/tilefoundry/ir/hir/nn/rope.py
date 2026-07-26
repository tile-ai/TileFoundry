"""Rotary Position Embedding (RoPE) HIR primitive.

SGLang baseline kernel K04. Applies position-dependent rotation to Q and K
along the last (head_dim) axis, using precomputed cos/sin caches indexed by
``pos_ids``.


Multi-output op: returns a tuple ``(q_rope, k_rope)``. Both share input shape /
dtype / layout / storage with their respective Q / K input.
"""
from __future__ import annotations

import isl
import torch

from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue, to_torch_dtype
from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.hir._shard_checks import check_multilinear_partials, reject_partials
from tilefoundry.ir.types import TupleType
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelationResult,
    AccessRelations,
    register_access_relation,
    register_type_relation,
)
from tilefoundry.visitor_registry.isl_utility import to_domain


@register_op
class RoPE(Op):
    """Rotary position embedding on Q and K. ``head_dim`` must be even."""
    q = ParamDef(kind="input", pattern=Tensor)
    k = ParamDef(kind="input", pattern=Tensor)
    cos_cache = ParamDef(kind="input", pattern=Tensor)
    sin_cache = ParamDef(kind="input", pattern=Tensor)
    pos_ids = ParamDef(kind="input", pattern=Tensor)


@register_typeinfer(RoPE)
def _(call: "Call", ctx: "TypeInferContext") -> TupleType:
    q_ty = ctx.type_of(call.args[0])
    k_ty = ctx.type_of(call.args[1])
    cos_ty = ctx.type_of(call.args[2])
    sin_ty = ctx.type_of(call.args[3])
    pos_ty = ctx.type_of(call.args[4])
    if not q_ty.shape or not k_ty.shape:
        ctx.error(call, "q and k must be at least rank-1")
    head_dim_q = q_ty.shape[-1]
    head_dim_k = k_ty.shape[-1]
    if isinstance(head_dim_q, int) and head_dim_q % 2 != 0:
        ctx.error(call, f"q head_dim {head_dim_q} must be even")
    if isinstance(head_dim_k, int) and head_dim_k % 2 != 0:
        ctx.error(call, f"k head_dim {head_dim_k} must be even")
    if (
        isinstance(head_dim_q, int)
        and isinstance(head_dim_k, int)
        and head_dim_q != head_dim_k
    ):
        ctx.error(call, f"q head_dim {head_dim_q} != k head_dim {head_dim_k}")
    # q*cos + rotate_half(q)*sin is multilinear in each value input. Each
    # output branch therefore allows one sum Partial (on q or k, the anchor
    # the output tuple preserves) only when the caches are replicated.
    check_multilinear_partials(
        ctx, call, (("q", q_ty), ("cos_cache", cos_ty), ("sin_cache", sin_ty)), anchor="q",
    )
    check_multilinear_partials(
        ctx, call, (("k", k_ty), ("cos_cache", cos_ty), ("sin_cache", sin_ty)), anchor="k",
    )
    # Indexed cache access does not commute with a partial (per-shard) pos_ids.
    reject_partials(ctx, call, "pos_ids", pos_ty.layout)
    return TupleType(fields=(q_ty, k_ty))

@register_access_relation(RoPE)
def _rope_access_relation(call: "Call", ctx: "TypeInferContext") -> AccessRelations:
    """GLOBAL level.

    Inputs:
      - q, k: per-element identity (rotation is per (token, head, head_dim/2 pair))
      - cos_cache, sin_cache: indexed by pos_ids → opaque (data-dependent index)
      - pos_ids: opaque (1D index input feeding cache lookup)

    Outputs:
      - q_rope, k_rope: per-element identity vs Q / K respectively.
    """
    q_ty = ctx.type_of(call.args[0])
    k_ty = ctx.type_of(call.args[1])

    def _ident(rank: int) -> "isl.multi_aff":
        dims = ", ".join(f"i{i}" for i in range(rank))
        return isl.multi_aff(f"{{ [{dims}] -> [{dims}] }}")

    q_id = _ident(len(q_ty.shape))
    k_id = _ident(len(k_ty.shape))

    return AccessRelations(
        inputs=(q_id, k_id, OPAQUE, OPAQUE, OPAQUE),
        outputs=(q_id, k_id),
    )

@register_type_relation(RoPE)
def _rope_type_relation(call: "Call", input_types, ctx) -> AccessRelationResult:
    """Forward relation for one RoPE branch: the value input paired with
    itself (``x, x, cos, sin, pos``) -- GQA's Hq != Hkv means q_rope and
    k_rope cannot share one domain, so ``analysis.poly``'s
    ``_rope_access`` calls this once per branch (q or k) and keeps only
    that branch's maps.

    cos_cache/sin_cache access is seq+head_dim identity, batch/head
    broadcast: V1 assumes prefill ``pos_ids == arange(seq)``, so
    ``cos_cache[pos_ids[s]] == cos_cache[s]`` -- the data-dependent gather
    degenerates to a plain seq-axis identity. pos_ids itself gets the same
    seq-identity access (decode's arbitrary pos_ids is a backlog item).
    """
    x_ty, x2_ty, cos_ty, sin_ty, pos_ty = input_types
    if x_ty.shape != x2_ty.shape:
        raise NotImplementedError(
            "RoPE type_relation: expects the value input paired with "
            "itself (analysis.extract splits a real Hq != Hkv call into "
            f"two such branches -- see _rope_access), got shapes "
            f"{x_ty.shape} vs {x2_ty.shape}"
        )
    if len(x_ty.shape) != 4:
        raise NotImplementedError(
            "RoPE type_relation: V1 only supports rank-4 [batch,seq,head,"
            f"head_dim] q/k, got shape {x_ty.shape}"
        )
    if len(cos_ty.shape) != 2 or len(sin_ty.shape) != 2:
        raise NotImplementedError(
            "RoPE type_relation: V1 only supports rank-2 [max_pos,head_dim] "
            f"cos/sin caches, got {cos_ty.shape} / {sin_ty.shape}"
        )

    domain, param_map = to_domain(x_ty.shape)
    x_map = isl.map("{ [d0,d1,d2,d3] -> [d0,d1,d2,d3] }")
    cache_map = isl.map("{ [d0,d1,d2,d3] -> [d1,d3] }")
    pos_map = isl.map("{ [d0,d1,d2,d3] -> [d1] }")
    # boundary order: q, k, cos_cache, sin_cache, pos_ids, q_rope, k_rope --
    # q/k slots and both outputs share x_map since x is paired with itself.
    maps = (x_map, x_map, cache_map, cache_map, pos_map, x_map, x_map)
    return AccessRelationResult(domain=domain, maps=maps, param_map=param_map)

@register_eval(RoPE)
def _eval_rope(ctx):
    # Layout is [batch, seq, head, head_dim]: cos/sin are gathered per token
    # from the caches by ``pos_ids`` and broadcast over the batch and head axes.
    # The rotation is the rotate-half form q*cos + rotate_half(q)*sin.
    q = ctx.args[0].data.float()
    k = ctx.args[1].data.float()
    pos = ctx.args[4].data.reshape(-1).long()
    cos = ctx.args[2].data[pos].float()[None, :, None, :]
    sin = ctx.args[3].data[pos].float()[None, :, None, :]

    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return TupleValue(
        elements=(
            TensorValue(
                data=q_out.to(to_torch_dtype(ctx.result_type.fields[0].dtype)),
                type=ctx.result_type.fields[0],
            ),
            TensorValue(
                data=k_out.to(to_torch_dtype(ctx.result_type.fields[1].dtype)),
                type=ctx.result_type.fields[1],
            ),
        )
    )


__all__ = ["RoPE"]
