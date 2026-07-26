"""Qwen2.5-1.5B dense decoder layer: pull a kernel by attribute, evaluate vs HF.

Phase 0 cpu + f32 oracle (no CUDA on this box — every ``device=`` below is
``"cpu"``). Each test resolves one kernel from the ``Qwen2_5_1_5B`` module
(mirroring ``tests/models/qwen3_1_7b/test_model/decoder_layer.py``) and checks
it against the corresponding Hugging Face ``Qwen2DecoderLayer`` submodule(s).
Inputs are built fresh inside each test from the shared ``common`` fixtures —
no module-level static tensors.
"""
from __future__ import annotations

import torch

from tests.models.qwen2_5_1_5b import config
from tests.models.qwen2_5_1_5b import qwen2_5_1_5b as model
from tilefoundry.evaluator import evaluate

HIDDEN = config.REAL.hidden
S_CAP = config.REAL.s_cap

DEV = "cpu"
ATOL = RTOL = 2e-4


def _fixtures():
    """A fresh HF layer + its RoPE caches / causal mask / attention scale, all
    on cpu. ``pos_ids`` is ``0..S_CAP-1`` — there is no prior KV-cache context
    in this package, so ``cur_pos`` is always 0 (see ``config.causal_mask``)."""
    layer = config.build_hf_layer(seed=0, device=DEV)
    cfg = config.build_hf_config()
    cos_cache, sin_cache = config.rope_caches(cfg, S_CAP, device=DEV)
    pos_ids = torch.arange(S_CAP, device=DEV, dtype=torch.int32)
    mask = config.causal_mask(S_CAP, device=DEV)
    scale = torch.full((1, 1, 1, 1), layer.self_attn.scaling, device=DEV)
    return layer, cos_cache, sin_cache, pos_ids, mask, scale


def test_input_rms_norm_evaluate():
    """input_rms_norm vs HF `input_layernorm`."""
    layer, *_ = _fixtures()
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    with torch.no_grad():
        ref = layer.input_layernorm(x)
    out = evaluate(model.input_rms_norm, x, layer.input_layernorm.weight, device=DEV)

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_self_attention_evaluate():
    """self_attention (input_layernorm + self_attn: GQA + RoPE + QKV bias, no
    q_norm/k_norm) vs HF, over a plain causal mask (cur_pos == 0, no prior
    KV-cache context in this package)."""
    layer, cos_cache, sin_cache, pos_ids, mask, scale = _fixtures()
    attn = layer.self_attn
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    cos = cos_cache[pos_ids.long()].unsqueeze(0)
    sin = sin_cache[pos_ids.long()].unsqueeze(0)
    with torch.no_grad():
        h = layer.input_layernorm(x)
        ref, _ = attn(h, position_embeddings=(cos, sin), attention_mask=mask)

    out = evaluate(
        model.self_attention,
        x,
        layer.input_layernorm.weight,
        config.linear_weight(attn.q_proj),
        attn.q_proj.bias,
        config.linear_weight(attn.k_proj),
        attn.k_proj.bias,
        config.linear_weight(attn.v_proj),
        attn.v_proj.bias,
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
    """Full decoder_layer (self_attention + residual + mlp + residual) vs the
    complete HF `Qwen2DecoderLayer.forward`."""
    layer, cos_cache, sin_cache, pos_ids, mask, scale = _fixtures()
    attn, mlp = layer.self_attn, layer.mlp
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    cos = cos_cache[pos_ids.long()].unsqueeze(0)
    sin = sin_cache[pos_ids.long()].unsqueeze(0)
    with torch.no_grad():
        ref = layer(x, position_embeddings=(cos, sin), attention_mask=mask)

    out = evaluate(
        model.decoder_layer,
        x,
        layer.input_layernorm.weight,
        config.linear_weight(attn.q_proj),
        attn.q_proj.bias,
        config.linear_weight(attn.k_proj),
        attn.k_proj.bias,
        config.linear_weight(attn.v_proj),
        attn.v_proj.bias,
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
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
