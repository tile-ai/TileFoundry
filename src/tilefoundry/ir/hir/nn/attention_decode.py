"""AttentionDecode big op (FA3-style decode with paged KV + GQA).

Combines drawio kernels K05 (FA3 prepare) + K06 (FlashAttention 3 decode) into
a single black-box HIR op. Internal structure (q·kᵀ → softmax → ·v, paged KV
read/write, varlen attention) is intentionally not exposed.

Two call modes are accepted to preserve backward compatibility:

- **Placeholder mode (3 args)**: ``AttentionDecode()`` with
  ``args=(q, k, v)``. Used by the legacy
  ``build_qwen3_attention_main_2cta_headnorm`` distribution candidate. typeinfer
  falls back to ``shape=batch + (num_q_heads*head_dim,)`` derived from q.
- **Full mode (7 args)**: ``args=(q, k, v, kv_cache, page_table, seq_lens, pos_ids)``
  + ``num_q_heads / num_kv_heads / head_dim`` attributes set explicitly. Used by
  the SGLang baseline graph.

"""
from __future__ import annotations

from tilefoundry.ir.core import Op
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.pattern import Tensor
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.core.registry import register_typeinfer
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard.shard_layout import partial_reductions
from tilefoundry.visitor_registry.access_relation import (
    OPAQUE,
    AccessRelations,
    register_access_relation,
)


@register_op(name="attention_decode")
class AttentionDecode(Op):
    """Decode-step attention with paged KV cache + GQA. Big op."""
    q = ParamDef(kind="input", pattern=Tensor)
    k = ParamDef(kind="input", pattern=Tensor)
    v = ParamDef(kind="input", pattern=Tensor)
    kv_cache = ParamDef(kind="input", pattern=Tensor)
    page_table = ParamDef(kind="input", pattern=Tensor)
    seq_lens = ParamDef(kind="input", pattern=Tensor)
    pos_ids = ParamDef(kind="input", pattern=Tensor)
    num_q_heads = ParamDef(kind="attribute", annotation=int, default=0)
    num_kv_heads = ParamDef(kind="attribute", annotation=int, default=0)
    head_dim = ParamDef(kind="attribute", annotation=int, default=0)
    page_size = ParamDef(kind="attribute", annotation=int, default=1)
    softmax_scale = ParamDef(kind="attribute", annotation=float, default=0.0)
@register_typeinfer(AttentionDecode)
def _(call: "Call", ctx: "TypeInferContext") -> TensorType:
    q_ty = ctx.type_of(call.args[0])
    target = call.target
    nargs = len(call.args)

    if nargs == 3:
        # Placeholder mode: derive shape from q (legacy path).
        if len(q_ty.shape) < 2:
            raise TypeError(
                "AttentionDecode placeholder mode: q must be rank ≥ 2 "
                f"(got shape {q_ty.shape})"
            )
        merged = q_ty.shape[-2] * q_ty.shape[-1]
        batch = q_ty.shape[:-2] if len(q_ty.shape) > 2 else (1,)
        out_shape = batch + (merged,)
    elif nargs == 7:
        # Full mode: shape derived from explicit attributes.
        if target.num_q_heads <= 0 or target.head_dim <= 0:
            raise TypeError(
                "AttentionDecode full mode requires num_q_heads > 0 and head_dim > 0"
            )
        if len(q_ty.shape) < 2:
            raise TypeError(
                "AttentionDecode: q must be rank ≥ 2"
            )
        # Fold (num_q_heads, head_dim) → (num_q_heads * head_dim).
        batch = q_ty.shape[:-2] if len(q_ty.shape) > 2 else (1,)
        out_shape = batch + (target.num_q_heads * target.head_dim,)
    else:
        raise TypeError(
            f"AttentionDecode: expected 3 (placeholder) or 7 (full) args, got {nargs}"
        )
    if partial_reductions(q_ty.layout):
        raise TypeError(
            "AttentionDecode: Partial input on q is unsound (an opaque "
            "black box containing softmax internally does not commute with "
            "any reduction) — insert reshard(q, Broadcast) before this "
            "consumer"
        )
    return TensorType(
        shape=out_shape,
        dtype=q_ty.dtype,
        layout=q_ty.layout,
        storage=q_ty.storage,
    )

@register_access_relation(AttentionDecode)
def _attention_decode_access_relation(
    call: "Call", ctx: "TypeInferContext"
) -> AccessRelations:
    """Big op — all boundaries OPAQUE per D12."""
    n_in = len(call.args)
    return AccessRelations(
        inputs=tuple(OPAQUE for _ in range(n_in)),
        outputs=(OPAQUE,),
    )

__all__ = ["AttentionDecode"]
