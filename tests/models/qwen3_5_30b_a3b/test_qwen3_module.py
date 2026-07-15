"""Qwen3-30B-A3B IR module: pull a kernel function and evaluate it vs HF.

The decoder is one ``Module`` (``Qwen3_30B_A3B``); each test resolves a single
kernel by attribute (``Qwen3_30B_A3B.self_attention``, mirroring the HF model)
and checks it against the Hugging Face reference. Inputs are constructed inside
each test from shape-field parameters (``cur_pos`` / ``s`` / dtype) — no
module-level static tensors.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.qwen3_5_30b_a3b import common
from tests.models.qwen3_5_30b_a3b.qwen3_module import Qwen3_30B_A3B
from tilefoundry.evaluator import evaluate

HEAD_DIM = common.HEAD_DIM
HIDDEN = common.HIDDEN
NUM_Q_HEADS = common.NUM_Q_HEADS
NUM_KV_HEADS = common.NUM_KV_HEADS
S_CAP = common.S_CAP
CACHE_CAP = common.CACHE_CAP


def _hf_qkv_rope_ref(attn, hidden_norm, cos, sin, pos_ids, k_cache0, v_cache0, cur_pos, s):
    """HF reference for the fused QkvRope kernel, starting from ``hidden_norm``
    (input RMSNorm is K1, kept separate): q/k/v projection (HF's separate
    ``q_proj``/``k_proj``/``v_proj`` are the unpacked form of the kernel's single
    packed ``w_qkv`` GEMM), per-head ``q_norm``/``k_norm``, RoPE, then scatter the
    first ``s`` rotated K / raw V tokens into the cache at ``cur_pos``."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    d = HEAD_DIM
    seq = hidden_norm.shape[1]
    cos_g = cos[pos_ids.long()].unsqueeze(0)  # [1, seq, d]
    sin_g = sin[pos_ids.long()].unsqueeze(0)
    # HF: proj -> view [1,seq,heads,d] -> per-head norm -> transpose to [1,heads,seq,d].
    q = attn.q_norm(attn.q_proj(hidden_norm).view(1, seq, NUM_Q_HEADS, d)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(hidden_norm).view(1, seq, NUM_KV_HEADS, d)).transpose(1, 2)
    v = attn.v_proj(hidden_norm).view(1, seq, NUM_KV_HEADS, d).transpose(1, 2)
    q, _ = apply_rotary_pos_emb(q, q, cos_g, sin_g)
    _, k = apply_rotary_pos_emb(k, k, cos_g, sin_g)
    q_rope = q.transpose(1, 2)  # [1, seq, Hq, d]
    k_rope = k.transpose(1, 2)  # [1, seq, Hkv, d]
    v_shd = v.transpose(1, 2)   # [1, seq, Hkv, d]

    k_cache1 = k_cache0.clone()
    v_cache1 = v_cache0.clone()
    k_cache1[:, cur_pos : cur_pos + s] = k_rope[:, :s]
    v_cache1[:, cur_pos : cur_pos + s] = v_shd[:, :s]
    return q_rope, k_cache1, v_cache1


_COMBOS = [(2, 1), (5, 3)]


@pytest.mark.parametrize("cur_pos,s", _COMBOS, ids=lambda v: str(v))
def test_qkv_rope_evaluate(cur_pos, s):
    """K2+K3+K4 fused QkvRope (packed GEMM + slice + per-head norm + RoPE + KV
    cache write), pulled from the module and checked against HF.

    The module is bf16 (per-op f32 numerics are covered in ``tests/ops``); this
    integration oracle checks structural correctness against HF at bf16."""
    torch_dt, atol, rtol = torch.bfloat16, 2e-2, 2e-2
    mod = Qwen3_30B_A3B
    fn = mod.qkv_rope

    dev = "cuda"
    layer = common.build_hf_layer(seed=0, device=dev, dtype=torch_dt)
    attn = layer.self_attn
    cfg = common.build_hf_config()
    cos, sin = common.rope_caches(cfg, CACHE_CAP, device=dev, dtype=torch_dt)
    pos_ids = torch.arange(cur_pos, cur_pos + S_CAP, device=dev, dtype=torch.int32)

    torch.manual_seed(1)
    hidden_norm = (torch.randn(1, S_CAP, HIDDEN, device=dev) * 0.1).to(torch_dt)
    # "preprocess weight": pack HF's separate q/k/v proj into one [1, HIDDEN,
    # QKV_FAN] GEMM weight (matmul convention is hidden_norm[1,S,in] @ w[1,in,out]).
    w_qkv = torch.cat(
        [attn.q_proj.weight.t(), attn.k_proj.weight.t(), attn.v_proj.weight.t()], dim=-1
    ).unsqueeze(0).contiguous()
    gamma_q = attn.q_norm.weight
    gamma_k = attn.k_norm.weight
    k_cache0 = (torch.randn(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    v_cache0 = (torch.randn(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    cur_pos_t = torch.tensor([cur_pos], device=dev, dtype=torch.int32)
    s_t = torch.tensor([s], device=dev, dtype=torch.int32)

    ref_q, ref_k, ref_v = _hf_qkv_rope_ref(
        attn, hidden_norm, cos, sin, pos_ids, k_cache0, v_cache0, cur_pos, s
    )

    q_rope, k_cache1, v_cache1 = evaluate(
        fn, hidden_norm, w_qkv, gamma_q, gamma_k, cos, sin, pos_ids,
        k_cache0, v_cache0, cur_pos_t, s_t, device=dev,
    )

    torch.testing.assert_close(q_rope.float(), ref_q.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(k_cache1.float(), ref_k.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(v_cache1.float(), ref_v.float(), atol=atol, rtol=rtol)


def test_input_rms_norm_evaluate():
    """K1 input RMSNorm, pulled from the module, vs HF ``input_layernorm``."""
    torch_dt, atol, rtol = torch.bfloat16, 2e-2, 2e-2
    mod = Qwen3_30B_A3B
    fn = mod.input_rms_norm

    dev = "cuda"
    torch.manual_seed(0)
    layer = common.build_hf_layer(seed=0, device=dev, dtype=torch_dt)
    x = (torch.randn(1, S_CAP, HIDDEN, device=dev) * 0.1).to(torch_dt)

    ref = layer.input_layernorm(x)
    out = evaluate(fn, x, layer.input_layernorm.weight, device=dev)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("cur_pos,s", _COMBOS, ids=lambda v: str(v))
def test_gqa_attend_evaluate(cur_pos, s):
    """K5+K6 masked GQA attention + output projection, pulled from the module,
    vs HF eager attention over the same q/kv tensors (inactive rows i >= s stay
    finite via the safe mask; only the valid rows are compared)."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa: PLC0415
        eager_attention_forward,
    )

    torch_dt, atol, rtol = torch.bfloat16, 2e-2, 2e-2
    mod = Qwen3_30B_A3B
    fn = mod.gqa_attend

    dev = "cuda"
    torch.manual_seed(0)
    layer = common.build_hf_layer(seed=0, device=dev, dtype=torch_dt)
    attn = layer.self_attn
    mask = common.decode_attn_mask(cur_pos, s, device=dev, dtype=torch_dt)
    scale = torch.full((1, 1, 1, 1), attn.scaling, device=dev, dtype=torch_dt)

    q_rope = (torch.randn(1, S_CAP, NUM_Q_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    k_cache = (torch.randn(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    v_cache = (torch.randn(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM, device=dev) * 0.1).to(torch_dt)
    w_o = attn.o_proj.weight.t().unsqueeze(0).contiguous()

    # HF reference: same tensors in HF layout [1, heads, seq, d]; eager attention
    # broadcasts kv heads and applies the scaling/mask, then output projection.
    attn_out, _ = eager_attention_forward(
        attn, q_rope.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2),
        mask, scaling=attn.scaling,
    )
    ref = attn.o_proj(attn_out.reshape(1, S_CAP, -1))

    out = evaluate(fn, q_rope, k_cache, v_cache, mask, scale, w_o, device=dev)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :s].float(), ref[:, :s].float(), atol=atol, rtol=rtol)


def _hf_self_attention_ref(attn, layer, x_real, k_cache0_hf, v_cache0_hf, cos, sin, mask_real, cur_pos, s):
    """HF decode-step self-attention (no residual): input RMSNorm, q/k/v proj +
    per-head norm + RoPE, scatter the new K/V into the cache, full-cache eager
    attention, output projection. Caches are HF layout [1, kv_heads, CACHE_CAP, d]."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (  # noqa: PLC0415
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

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


@pytest.mark.parametrize("cur_pos,s", _COMBOS, ids=lambda v: str(v))
def test_self_attention_evaluate(cur_pos, s):
    """Composed decode-step self-attention (input_rms_norm -> qkv_rope ->
    gqa_attend), pulled from the module by attribute, vs HF. Inactive padding
    rows stay finite; only the valid rows and the updated caches are compared."""
    torch_dt, atol, rtol = torch.bfloat16, 2e-2, 2e-2
    mod = Qwen3_30B_A3B
    fn = mod.self_attention

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

    cos = cos_cache[pos_ids[:s].long()].unsqueeze(0)
    sin = sin_cache[pos_ids[:s].long()].unsqueeze(0)
    ref_out, ref_k, ref_v = _hf_self_attention_ref(
        attn, layer, x[:, :s], k_cache0.transpose(1, 2), v_cache0.transpose(1, 2),
        cos, sin, mask[:, :, :s, :], cur_pos, s,
    )

    gamma_in = layer.input_layernorm.weight
    w_qkv = torch.cat(
        [attn.q_proj.weight.t(), attn.k_proj.weight.t(), attn.v_proj.weight.t()], dim=-1
    ).unsqueeze(0).contiguous()
    w_o = attn.o_proj.weight.t().unsqueeze(0).contiguous()

    out, k_cache1, v_cache1 = evaluate(
        fn, x, gamma_in, w_qkv, attn.q_norm.weight, attn.k_norm.weight,
        cos_cache, sin_cache, pos_ids, k_cache0, v_cache0, cur_pos_t, s_t, mask, scale, w_o,
        device=dev,
    )

    assert torch.isfinite(out).all()
    torch.testing.assert_close(k_cache1.float(), ref_k.transpose(1, 2).float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(v_cache1.float(), ref_v.transpose(1, 2).float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(out[:, :s].float(), ref_out.float(), atol=atol, rtol=rtol)
