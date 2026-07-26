"""MiniCPM3-4B single decoder layer: pull a kernel by attribute, evaluate vs
HF. Phase 0 cpu + f32 oracle (no CUDA on this box — every ``device=`` below
is ``"cpu"``). Each test resolves one kernel from the ``MiniCPM3_4B`` module
(mirroring ``tests/models/qwen3_1_7b/test_model/decoder_layer.py``) and checks
it against the corresponding Hugging Face ``MiniCPM3DecoderLayer``
submodule(s). Inputs are built fresh inside each test from the shared
``common`` fixtures — no module-level static tensors.
"""
from __future__ import annotations

import torch

from tests.models.minicpm3_4b import config
from tests.models.minicpm3_4b import minicpm3_4b as model
from tilefoundry.evaluator import evaluate

HIDDEN = config.REAL.hidden
S_CAP = config.REAL.s_cap

DEV = "cpu"
ATOL = RTOL = 2e-4


def _fixtures():
    """A fresh HF layer + its RoPE caches / causal mask / attention scale /
    residual scale, all on cpu. ``pos_ids`` is ``0..S_CAP-1`` — there is no
    prior KV-cache context in this package, so ``cur_pos`` is always 0 (see
    ``config.causal_mask``). ``residual_scale`` is read directly off the HF
    layer's own precomputed ``scale_depth / sqrt(num_hidden_layers)``, not
    re-derived from config fields, so this test has no independent copy of
    that formula to drift out of sync."""
    layer = config.build_hf_layer(seed=0, device=DEV)
    cfg = config.build_hf_config()
    cos_cache, sin_cache = config.rope_caches(cfg, S_CAP, device=DEV)
    pos_ids = torch.arange(S_CAP, device=DEV, dtype=torch.int32)
    mask = config.causal_mask(S_CAP, device=DEV)
    scale = torch.full((1, 1, 1, 1), layer.self_attn.scaling, device=DEV)
    residual_scale = torch.full((1, 1, 1), layer.residual_scale, device=DEV)
    return layer, cos_cache, sin_cache, pos_ids, mask, scale, residual_scale


def test_input_rms_norm_evaluate():
    """input_rms_norm vs HF `input_layernorm`."""
    layer, *_ = _fixtures()
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    with torch.no_grad():
        ref = layer.input_layernorm(x)
    out = evaluate(model.input_rms_norm, x, layer.input_layernorm.weight, device=DEV)

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_mla_attention_evaluate():
    """mla_attention (input_layernorm + MLA self_attn: low-rank Q, shared
    low-rank KV latent, rope-slice-only RoPE, MQA-shared k_rope) vs HF, over
    a plain causal mask (cur_pos == 0, no prior KV-cache context)."""
    layer, cos_cache, sin_cache, pos_ids, mask, scale, _ = _fixtures()
    attn = layer.self_attn
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    cos = cos_cache[pos_ids.long()].unsqueeze(0)
    sin = sin_cache[pos_ids.long()].unsqueeze(0)
    with torch.no_grad():
        h = layer.input_layernorm(x)
        ref, _ = attn(h, position_embeddings=(cos, sin), attention_mask=mask)

    out = evaluate(
        model.mla_attention,
        x,
        layer.input_layernorm.weight,
        config.linear_weight(attn.q_a_proj),
        attn.q_a_layernorm.weight,
        config.linear_weight(attn.q_b_proj),
        config.linear_weight(attn.kv_a_proj_with_mqa),
        attn.kv_a_layernorm.weight,
        config.linear_weight(attn.kv_b_proj),
        cos_cache,
        sin_cache,
        pos_ids,
        mask,
        scale,
        config.linear_weight(attn.o_proj),
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_mlp_evaluate():
    """mlp (post_attention_layernorm + dense SwiGLU) vs HF."""
    layer, *_ = _fixtures()
    mlp = layer.mlp
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    with torch.no_grad():
        ref = mlp(layer.post_attention_layernorm(x))

    out = evaluate(
        model.mlp,
        x,
        layer.post_attention_layernorm.weight,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_decoder_layer_evaluate():
    """Full decoder_layer (mla_attention + scale_depth residual + mlp +
    scale_depth residual) vs the complete HF `MiniCPM3DecoderLayer.forward`
    — the scale_depth residual scaling is the one place this decoder_layer
    diverges structurally from the Qwen3-1.7B sibling (plain-add residual)."""
    layer, cos_cache, sin_cache, pos_ids, mask, scale, residual_scale = _fixtures()
    attn, mlp = layer.self_attn, layer.mlp
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    cos = cos_cache[pos_ids.long()].unsqueeze(0)
    sin = sin_cache[pos_ids.long()].unsqueeze(0)
    with torch.no_grad():
        ref = layer(x, attention_mask=mask, position_embeddings=(cos, sin))

    out = evaluate(
        model.decoder_layer,
        x,
        layer.input_layernorm.weight,
        config.linear_weight(attn.q_a_proj),
        attn.q_a_layernorm.weight,
        config.linear_weight(attn.q_b_proj),
        config.linear_weight(attn.kv_a_proj_with_mqa),
        attn.kv_a_layernorm.weight,
        config.linear_weight(attn.kv_b_proj),
        cos_cache,
        sin_cache,
        pos_ids,
        mask,
        scale,
        config.linear_weight(attn.o_proj),
        layer.post_attention_layernorm.weight,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
        residual_scale,
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
