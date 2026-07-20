"""Qwen3-30B-A3B decode-step: fixed-capacity cache HIR + evaluator-vs-HF.

A single-Function decode-step (cache write + read together) over a static
``CACHE_CAP`` KV cache: ``cache_update`` writes the new K/V at ``cur_pos`` and
the attention reads the full cache + mask. Validated against the HF
``Qwen3MoeDecoderLayer`` submodules, applying the same torch cache update first.
All decode-step tensor shapes are static; dynamism lives only in the runtime
scalars ``cur_pos`` / ``s`` and the boundary ``pos_ids`` / ``mask``.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.qwen3_5_30b_a3b import common
from tilefoundry.evaluator import evaluate

HIDDEN = common.HIDDEN
HEAD_DIM = common.HEAD_DIM
NUM_KV_HEADS = common.NUM_KV_HEADS
MOE_INTERMEDIATE = common.MOE_INTERMEDIATE
S_CAP = common.S_CAP
CACHE_CAP = common.CACHE_CAP


def _hf_decode_ref(layer, x_real, k_cache0_hf, v_cache0_hf, cos, sin, mask_real, cur_pos, s):
    """HF decode reference: same torch cache update, then full-cache attention.

    ``x_real`` is ``[1, s, H]``; caches are HF layout ``[1, kv_heads, CACHE_CAP,
    D]``; returns ``(out[1, s, H], k_cache1, v_cache1)``.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa: PLC0415
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    attn = layer.self_attn
    d = attn.head_dim
    h = layer.input_layernorm(x_real)
    q = attn.q_norm(attn.q_proj(h).view(1, s, -1, d)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(h).view(1, s, -1, d)).transpose(1, 2)
    v = attn.v_proj(h).view(1, s, -1, d).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, cos, sin)
    _, k = apply_rotary_pos_emb(k, k, cos, sin)
    k_cache1 = k_cache0_hf.clone()
    v_cache1 = v_cache0_hf.clone()
    k_cache1[:, :, cur_pos : cur_pos + s] = k
    v_cache1[:, :, cur_pos : cur_pos + s] = v
    attn_out, _ = eager_attention_forward(attn, q, k_cache1, v_cache1, mask_real, scaling=attn.scaling)
    out = attn.o_proj(attn_out.reshape(1, s, -1))
    return out, k_cache1, v_cache1


def _hf_decode_layer_ref(layer, x_real, k_cache0_hf, v_cache0_hf, cos, sin, mask_real, cur_pos, s):
    """HF full decode-step layer: attention + residual + MoE + residual.

    Returns ``(out[1, s, H], k_cache1, v_cache1)`` for the ``s`` real tokens,
    mirroring ``Qwen3MoeDecoderLayer.forward`` (the two layer-level residuals
    wrap the attention and the MoE block respectively).
    """
    attn_out, k_cache1, v_cache1 = _hf_decode_ref(
        layer, x_real, k_cache0_hf, v_cache0_hf, cos, sin, mask_real, cur_pos, s
    )
    h1 = x_real + attn_out
    moe_out = layer.mlp(layer.post_attention_layernorm(h1))
    out = h1 + moe_out
    return out, k_cache1, v_cache1


_DTYPES = [("f32", torch.float32, 2e-4, 2e-4), ("bf16", torch.bfloat16, 2e-2, 2e-2)]


@pytest.mark.parametrize(
    "dt_name,torch_dt,atol,rtol", _DTYPES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_decode_attention_matches_hf(dt_name, torch_dt, atol, rtol):
    cur_pos, s = 5, 3
    common.DT = dt_name
    fn = common.build_decode_attention()

    dev = "cuda"
    torch.manual_seed(0)
    layer = common.build_hf_layer(seed=0, device=dev, dtype=torch_dt)
    attn = layer.self_attn

    cos_cache, sin_cache = common.rope_caches(attn.config, CACHE_CAP, device=dev, dtype=torch_dt)
    pos_ids = torch.arange(cur_pos, cur_pos + S_CAP, device=dev, dtype=torch.int32)
    mask = common.decode_attn_mask(cur_pos, s, device=dev, dtype=torch_dt)
    scale = torch.full((1, 1, 1, 1), attn.scaling, device=dev, dtype=torch_dt)

    x = (torch.randn(1, S_CAP, HIDDEN, device=dev) * 0.1).to(torch_dt)
    k_cache0 = (torch.randn(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    v_cache0 = (torch.randn(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    cur_pos_t = torch.tensor([cur_pos], device=dev, dtype=torch.int32)
    s_t = torch.tensor([s], device=dev, dtype=torch.int32)

    # HF reference over the s real tokens.
    cos = cos_cache[pos_ids[:s].long()].unsqueeze(0)
    sin = sin_cache[pos_ids[:s].long()].unsqueeze(0)
    ref_out, ref_k, ref_v = _hf_decode_ref(
        layer, x[:, :s], k_cache0.transpose(1, 2), v_cache0.transpose(1, 2),
        cos, sin, mask[:, :, :s, :], cur_pos, s,
    )

    gamma_in = layer.input_layernorm.weight
    w_q = attn.q_proj.weight.t().unsqueeze(0).contiguous()
    w_k = attn.k_proj.weight.t().unsqueeze(0).contiguous()
    w_v = attn.v_proj.weight.t().unsqueeze(0).contiguous()
    w_o = attn.o_proj.weight.t().unsqueeze(0).contiguous()

    out, k_cache1, v_cache1 = evaluate(
        fn, x, gamma_in, w_q, w_k, w_v, attn.q_norm.weight, attn.k_norm.weight,
        cos_cache, sin_cache, pos_ids, k_cache0, v_cache0, cur_pos_t, s_t, mask, scale, w_o,
        device=dev,
    )

    # Inactive padding rows (i >= s) stay finite — the safe mask key avoids NaN.
    assert torch.isfinite(out).all()
    # Updated caches match (IR layout [1, CACHE_CAP, kv_heads, D] vs HF [1, kv_heads, CACHE_CAP, D]).
    torch.testing.assert_close(k_cache1.float(), ref_k.transpose(1, 2).float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(v_cache1.float(), ref_v.transpose(1, 2).float(), atol=atol, rtol=rtol)
    # Valid output rows match.
    torch.testing.assert_close(out[:, :s].float(), ref_out.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dt_name,torch_dt,atol,rtol", _DTYPES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_decode_layer_matches_hf(dt_name, torch_dt, atol, rtol):
    """Full decode-step layer (one composed Function) vs HF Qwen3MoeDecoderLayer."""
    cur_pos, s = 5, 3
    common.DT = dt_name
    fn = common.build_decode_layer()

    dev = "cuda"
    torch.manual_seed(0)
    layer = common.build_hf_layer(seed=0, device=dev, dtype=torch_dt)
    attn = layer.self_attn

    cos_cache, sin_cache = common.rope_caches(attn.config, CACHE_CAP, device=dev, dtype=torch_dt)
    pos_ids = torch.arange(cur_pos, cur_pos + S_CAP, device=dev, dtype=torch.int32)
    mask = common.decode_attn_mask(cur_pos, s, device=dev, dtype=torch_dt)
    scale = torch.full((1, 1, 1, 1), attn.scaling, device=dev, dtype=torch_dt)

    x = (torch.randn(1, S_CAP, HIDDEN, device=dev) * 0.1).to(torch_dt)
    k_cache0 = (torch.randn(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    v_cache0 = (torch.randn(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    cur_pos_t = torch.tensor([cur_pos], device=dev, dtype=torch.int32)
    s_t = torch.tensor([s], device=dev, dtype=torch.int32)

    # HF reference over the s real tokens.
    cos = cos_cache[pos_ids[:s].long()].unsqueeze(0)
    sin = sin_cache[pos_ids[:s].long()].unsqueeze(0)
    ref_out, ref_k, ref_v = _hf_decode_layer_ref(
        layer, x[:, :s], k_cache0.transpose(1, 2), v_cache0.transpose(1, 2),
        cos, sin, mask[:, :, :s, :], cur_pos, s,
    )

    gamma_in = layer.input_layernorm.weight
    w_q = attn.q_proj.weight.t().unsqueeze(0).contiguous()
    w_k = attn.k_proj.weight.t().unsqueeze(0).contiguous()
    w_v = attn.v_proj.weight.t().unsqueeze(0).contiguous()
    w_o = attn.o_proj.weight.t().unsqueeze(0).contiguous()

    moe = layer.mlp
    gup = moe.experts.gate_up_proj
    w_gate = gup[:, :MOE_INTERMEDIATE, :].contiguous()
    w_up = gup[:, MOE_INTERMEDIATE:, :].contiguous()
    w_down = moe.experts.down_proj.contiguous()
    w_router = moe.gate.weight.t().contiguous()

    out, k_cache1, v_cache1 = evaluate(
        fn, x, gamma_in, w_q, w_k, w_v, attn.q_norm.weight, attn.k_norm.weight,
        cos_cache, sin_cache, pos_ids, k_cache0, v_cache0, cur_pos_t, s_t, mask, scale, w_o,
        layer.post_attention_layernorm.weight, w_router, w_gate, w_up, w_down,
        device=dev,
    )

    # Inactive padding rows (i >= s) stay finite — the safe mask key avoids NaN.
    assert torch.isfinite(out).all()
    # Updated caches match (IR layout [1, CACHE_CAP, kv_heads, D] vs HF [1, kv_heads, CACHE_CAP, D]).
    torch.testing.assert_close(k_cache1.float(), ref_k.transpose(1, 2).float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(v_cache1.float(), ref_v.transpose(1, 2).float(), atol=atol, rtol=rtol)
    # Valid output rows match the full HF decoder-layer output.
    torch.testing.assert_close(out[:, :s].float(), ref_out.float(), atol=atol, rtol=rtol)
