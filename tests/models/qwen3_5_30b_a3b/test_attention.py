"""Qwen3-30B-A3B attention component: bf16 HIR description + evaluator-vs-HF.

The attention math is validated as two HIR ``@func``s (built in ``common``)
rather than one composed Function: the IR relation engine cannot recover a
compound ``DimVar`` axis (``prior + new`` from a ``concat``) when it feeds
``matmul`` / ``transpose``, so the KV-cache append and the score computation
cannot live in one Function. They are validated separately and then chained
through Python against the ``input_layernorm`` + ``self_attn`` of a
Qwen3-30B-A3B ``Qwen3MoeDecoderLayer`` (the residual add belongs to the layer).
"""
from __future__ import annotations

import pytest
import torch

from tests.models.qwen3_5_30b_a3b import common
from tilefoundry.evaluator import evaluate

HIDDEN = common.HIDDEN
HEAD_DIM = common.HEAD_DIM
NUM_KV_HEADS = common.NUM_KV_HEADS


def _hf_pieces(layer, x, k_prev_hf, v_prev_hf, cos, sin, mask):
    """HF reference: the roped+appended full cache and the attention output of
    ``input_layernorm -> self_attn`` (no residual), prior K/V prepended."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa: PLC0415
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    attn = layer.self_attn
    b, s, _ = x.shape
    d = attn.head_dim
    h = layer.input_layernorm(x)
    q = attn.q_norm(attn.q_proj(h).view(b, s, -1, d)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(h).view(b, s, -1, d)).transpose(1, 2)
    v = attn.v_proj(h).view(b, s, -1, d).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    k_full = torch.cat([k_prev_hf, k], dim=2)
    v_full = torch.cat([v_prev_hf, v], dim=2)
    attn_out, _ = eager_attention_forward(attn, q, k_full, v_full, mask, scaling=attn.scaling)
    out = attn.o_proj(attn_out.reshape(b, s, -1))
    return k_full, v_full, out


_DTYPES = [("f32", torch.float32, 2e-4, 2e-4), ("bf16", torch.bfloat16, 2e-2, 2e-2)]


@pytest.mark.parametrize(
    "dt_name,torch_dt,atol,rtol", _DTYPES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_attention_matches_hf(dt_name, torch_dt, atol, rtol):
    seq, cur_pos = 3, 5
    common.DT = dt_name
    kv_update = common.build_kv_update()
    scores = common.build_scores()

    dev = "cuda"
    torch.manual_seed(0)
    ctx = cur_pos + seq
    layer = common.build_hf_layer(seed=0, device=dev, dtype=torch_dt)
    attn = layer.self_attn

    cos_cache, sin_cache = common.rope_caches(attn.config, ctx + 4, device=dev, dtype=torch_dt)
    pos_ids = torch.arange(cur_pos, cur_pos + seq, device=dev, dtype=torch.int32)
    mask = common.additive_causal_mask(seq, cur_pos, ctx, device=dev, dtype=torch_dt)

    x = (torch.randn(1, seq, HIDDEN, device=dev) * 0.1).to(torch_dt)
    k_prev = (torch.randn(1, cur_pos, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    v_prev = (torch.randn(1, cur_pos, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    scale = torch.full((1, 1, 1, 1), attn.scaling, device=dev, dtype=torch_dt)

    cos = cos_cache[pos_ids.long()].unsqueeze(0)
    sin = sin_cache[pos_ids.long()].unsqueeze(0)
    k_full_hf, v_full_hf, ref = _hf_pieces(
        layer, x, k_prev.transpose(1, 2), v_prev.transpose(1, 2), cos, sin, mask
    )

    gamma_in = layer.input_layernorm.weight
    w_k = attn.k_proj.weight.t().unsqueeze(0).contiguous()
    w_v = attn.v_proj.weight.t().unsqueeze(0).contiguous()
    w_q = attn.q_proj.weight.t().unsqueeze(0).contiguous()
    w_o = attn.o_proj.weight.t().unsqueeze(0).contiguous()

    k_full_ir, v_full_ir = evaluate(
        kv_update, x, gamma_in, w_k, w_v, attn.k_norm.weight,
        cos_cache, sin_cache, pos_ids, k_prev, v_prev, device=dev,
    )
    # IR cache layout [1, ctx, kv_heads, head_dim] vs HF [1, kv_heads, ctx, head_dim].
    torch.testing.assert_close(
        k_full_ir.float(), k_full_hf.transpose(1, 2).float(), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        v_full_ir.float(), v_full_hf.transpose(1, 2).float(), atol=atol, rtol=rtol
    )

    out = evaluate(
        scores, x, gamma_in, w_q, attn.q_norm.weight,
        cos_cache, sin_cache, pos_ids, k_full_ir, v_full_ir, mask, scale, w_o,
        device=dev,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
