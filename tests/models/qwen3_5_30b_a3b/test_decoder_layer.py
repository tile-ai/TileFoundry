"""Qwen3-30B-A3B decoder layer: bf16 HIR composition + evaluator-vs-HF.

Chains the attention (``kv_update`` -> ``scores``) and MoE component ``@func``s
with layer-level residual adds (``x + attn``, ``h + moe``) and validates the
result against a complete Qwen3-30B-A3B ``Qwen3MoeDecoderLayer``. Per the
compound-``DimVar`` limitation the stages are evaluated separately and chained
through Python — this is a chained validation, not one composed Function.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.qwen3_5_30b_a3b import common
from tests.models.qwen3_5_30b_a3b.test_attention import _hf_pieces
from tilefoundry.evaluator import evaluate

HIDDEN = common.HIDDEN
HEAD_DIM = common.HEAD_DIM
NUM_KV_HEADS = common.NUM_KV_HEADS
MOE_INTERMEDIATE = common.MOE_INTERMEDIATE


_DTYPES = [("f32", torch.float32, 1e-3, 1e-3), ("bf16", torch.bfloat16, 3e-2, 3e-2)]


@pytest.mark.parametrize(
    "dt_name,torch_dt,atol,rtol", _DTYPES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_decoder_layer_matches_hf(dt_name, torch_dt, atol, rtol):
    seq, cur_pos = 3, 5
    common.DT = dt_name
    kv_update = common.build_kv_update()
    scores = common.build_scores()
    moe = common.build_moe()
    residual = common.build_residual()

    dev = "cuda"
    torch.manual_seed(0)
    ctx = cur_pos + seq
    layer = common.build_hf_layer(seed=0, device=dev, dtype=torch_dt)
    attn = layer.self_attn
    moe_mod = layer.mlp

    cos_cache, sin_cache = common.rope_caches(attn.config, ctx + 4, device=dev, dtype=torch_dt)
    pos_ids = torch.arange(cur_pos, cur_pos + seq, device=dev, dtype=torch.int32)
    mask = common.additive_causal_mask(seq, cur_pos, ctx, device=dev, dtype=torch_dt)

    x = (torch.randn(1, seq, HIDDEN, device=dev) * 0.1).to(torch_dt)
    k_prev = (torch.randn(1, cur_pos, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    v_prev = (torch.randn(1, cur_pos, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    scale = torch.full((1, 1, 1, 1), attn.scaling, device=dev, dtype=torch_dt)

    cos = cos_cache[pos_ids.long()].unsqueeze(0)
    sin = sin_cache[pos_ids.long()].unsqueeze(0)

    # HF reference: the decoder layer forward over its submodules.
    with torch.no_grad():
        _, _, attn_out_ref = _hf_pieces(
            layer, x, k_prev.transpose(1, 2), v_prev.transpose(1, 2), cos, sin, mask
        )
        h1_ref = x + attn_out_ref
        ref = h1_ref + moe_mod(layer.post_attention_layernorm(h1_ref))

    gamma_in = layer.input_layernorm.weight
    w_q = attn.q_proj.weight.t().unsqueeze(0).contiguous()
    w_k = attn.k_proj.weight.t().unsqueeze(0).contiguous()
    w_v = attn.v_proj.weight.t().unsqueeze(0).contiguous()
    w_o = attn.o_proj.weight.t().unsqueeze(0).contiguous()
    gup = moe_mod.experts.gate_up_proj
    w_gate = gup[:, :MOE_INTERMEDIATE, :].contiguous()
    w_up = gup[:, MOE_INTERMEDIATE:, :].contiguous()
    w_down = moe_mod.experts.down_proj.contiguous()
    w_router = moe_mod.gate.weight.t().contiguous()

    k_full, v_full = evaluate(
        kv_update, x, gamma_in, w_k, w_v, attn.k_norm.weight,
        cos_cache, sin_cache, pos_ids, k_prev, v_prev, device=dev,
    )
    attn_out = evaluate(
        scores, x, gamma_in, w_q, attn.q_norm.weight,
        cos_cache, sin_cache, pos_ids, k_full, v_full, mask, scale, w_o, device=dev,
    )
    h1 = evaluate(residual, x, attn_out, device=dev)
    moe_out = evaluate(
        moe, h1, layer.post_attention_layernorm.weight, w_router, w_gate, w_up, w_down,
        device=dev,
    )
    out = evaluate(residual, h1, moe_out, device=dev)

    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
