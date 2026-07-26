"""Gemma-2-2B single decoder layer: pull a kernel by attribute, evaluate vs HF.

Phase 0 cpu + f32 oracle (no CUDA on this box — every ``device=`` below is
``"cpu"``). Each test resolves one kernel from the ``Gemma2_2B`` module
(mirroring ``tests/models/qwen3_1_7b/test_model/decoder_layer.py``) and checks it
against the corresponding Hugging Face ``Gemma2DecoderLayer`` submodule(s).
Inputs are built fresh inside each test from the shared ``common`` fixtures —
no module-level static tensors.

``self_attention`` / ``mlp`` are pure blocks here (unlike qwen3_1_7b, which
fuses each block's preceding norm) — see ``model/decoder_layer.py``'s docstring
for why Gemma-2's four-norm layout makes that the natural split. So their
tests apply the HF norm in plain torch test code before calling the kernel,
rather than threading a `gamma_*` kernel argument through.
"""
from __future__ import annotations

import torch

from tests.models.gemma2_2b import config
from tests.models.gemma2_2b import gemma2_2b as model
from tilefoundry.evaluator import evaluate

HIDDEN = config.REAL.hidden
S_CAP = config.REAL.s_cap

DEV = "cpu"
ATOL = RTOL = 2e-4


def _fixtures():
    """A fresh HF layer + its RoPE caches / causal mask / attention scale, all
    on cpu. ``pos_ids`` is ``0..S_CAP-1`` — there is no prior KV-cache context
    in this package, so ``cur_pos`` is always 0 (see ``config.causal_mask``).
    ``scale`` is ``query_pre_attn_scalar**-0.5`` (0.0625 @ 256) — Gemma-2 does
    NOT use ``head_dim**-0.5``."""
    layer = config.build_hf_layer(seed=0, device=DEV)
    cfg = config.build_hf_config()
    cos_cache, sin_cache = config.rope_caches(cfg, S_CAP, device=DEV)
    pos_ids = torch.arange(S_CAP, device=DEV, dtype=torch.int32)
    mask = config.causal_mask(S_CAP, device=DEV)
    scale = torch.full((1, 1, 1, 1), layer.self_attn.scaling, device=DEV)
    return layer, cos_cache, sin_cache, pos_ids, mask, scale


def test_input_rms_norm_evaluate():
    """input_rms_norm vs HF `input_layernorm` — `Gemma2RMSNorm` is
    `normed * (1.0 + weight)`, so `gamma_in` is pre-adjusted via
    `config.rms_gamma`."""
    layer, *_ = _fixtures()
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    with torch.no_grad():
        ref = layer.input_layernorm(x)
    out = evaluate(
        model.input_rms_norm, x, config.rms_gamma(layer.input_layernorm), device=DEV,
    )

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_self_attention_evaluate():
    """self_attention (pure GQA + RoPE + query_pre_attn_scalar scaling +
    attn_logit_softcapping) vs HF `input_layernorm` -> `self_attn`, over a
    plain causal mask (cur_pos == 0, no prior KV-cache context)."""
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
        h,
        config.linear_weight(attn.q_proj),
        config.linear_weight(attn.k_proj),
        config.linear_weight(attn.v_proj),
        cos_cache,
        sin_cache,
        pos_ids,
        mask,
        scale,
        config.linear_weight(attn.o_proj),
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_self_attention_sliding_window_evaluate():
    """Optional extra coverage (hint #6): a hand-built causal+sliding-window
    mask (window=2, small enough to actually differ from plain causal at
    S_CAP=4) fed as `attn_mask`. HF's `eager_attention_forward` only ever adds
    whatever `attention_mask` tensor it is given — Gemma-2's real
    sliding-window restriction lives upstream, in
    `create_sliding_window_causal_mask` (never consulted when calling
    `self_attn` directly, as every test in this file does) — so feeding the
    identical hand-built mask to both HIR and HF is a faithful, self-
    consistent oracle for the masking logic itself, independent of whether
    `SLIDING_WINDOW` (4096) would ever bind at this S_CAP."""
    layer, cos_cache, sin_cache, pos_ids, _mask, scale = _fixtures()
    attn = layer.self_attn
    sliding_mask = config.sliding_causal_mask(S_CAP, window=2, device=DEV)
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    cos = cos_cache[pos_ids.long()].unsqueeze(0)
    sin = sin_cache[pos_ids.long()].unsqueeze(0)
    with torch.no_grad():
        h = layer.input_layernorm(x)
        ref, _ = attn(h, position_embeddings=(cos, sin), attention_mask=sliding_mask)

    out = evaluate(
        model.self_attention,
        h,
        config.linear_weight(attn.q_proj),
        config.linear_weight(attn.k_proj),
        config.linear_weight(attn.v_proj),
        cos_cache,
        sin_cache,
        pos_ids,
        sliding_mask,
        scale,
        config.linear_weight(attn.o_proj),
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_mlp_gelu_tanh_evaluate():
    """mlp (pure dense gelu_pytorch_tanh-gated block) vs HF `Gemma2MLP`
    directly — no norm on either side (`hidden_activation="gelu_pytorch_tanh"`,
    not SwiGLU's `silu`; this is the one new `src/` op this package adds).
    Named so ``pytest tests/ -k gelu`` selects it as a standalone proof the
    new ``Gelu`` op composes and evaluates correctly."""
    layer, *_ = _fixtures()
    mlp = layer.mlp
    torch.manual_seed(1)
    x = torch.randn(1, S_CAP, HIDDEN, device=DEV) * 0.1

    with torch.no_grad():
        ref = mlp(x)

    out = evaluate(
        model.mlp,
        x,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_decoder_layer_evaluate():
    """Full decoder_layer (4 norms + 2 residual adds:
    `h = x + post_attn_norm(attn(input_norm(x)))`;
    `out = h + post_ff_norm(mlp(pre_ff_norm(h)))`) vs the complete HF
    `Gemma2DecoderLayer.forward`."""
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
        config.rms_gamma(layer.input_layernorm),
        config.linear_weight(attn.q_proj),
        config.linear_weight(attn.k_proj),
        config.linear_weight(attn.v_proj),
        cos_cache,
        sin_cache,
        pos_ids,
        mask,
        scale,
        config.linear_weight(attn.o_proj),
        config.rms_gamma(layer.post_attention_layernorm),
        config.rms_gamma(layer.pre_feedforward_layernorm),
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
        config.rms_gamma(layer.post_feedforward_layernorm),
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
