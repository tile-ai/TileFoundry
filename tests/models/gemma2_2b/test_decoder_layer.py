"""Gemma-2-2B decode step: resolve a kernel by name, evaluate vs HF.

cpu + f32 oracle. Each test resolves one kernel from the ``Gemma2_2B`` module and
checks it against the corresponding Hugging Face ``Gemma2DecoderLayer``
submodule(s).

The kernels that read the KV cache carry ``ctx_len`` as a range, so they are
specialised at the length the drawn step uses before being evaluated: an extent
is what counting elements needs, and a range is not one. The kernels that do not
read the cache carry no range and are evaluated as authored.

Arguments come from ``reference.py``'s drawn step rather than being assembled
here, so the parameter order is stated once and a signature change cannot leave
one test agreeing with a stale order. ``self_attention`` and ``mlp`` are pure
blocks (see ``model/decoder_layer.py``'s docstring for why Gemma-2's four-norm
layout makes that the natural split), so their tests apply the HF norm in plain
torch before calling the kernel.
"""
from __future__ import annotations

import torch

from tests.models import decode_oracle as oracle
from tests.models.gemma2_2b import config, reference
from tests.models.gemma2_2b import gemma2_2b as model
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.specialize import specialize_concretely

HIDDEN = config.REAL.hidden
SEQ = config.SEQ_LEN

DEV = "cpu"
ATOL = RTOL = 2e-4

#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither divides the key/value head count.
CTX_LENGTHS = (24, 40)

#: How much the drawn query is scaled up to make `attn_logit_softcapping` bite on
#: the new token's single logit rather than only on the cache's many.
SOFTCAP_PROBE_SCALE = 100.0


def _one_token(seed=1):
    """A fresh HF layer and one token's hidden states."""
    layer = config.build_hf_layer(seed=0, device=DEV)
    torch.manual_seed(seed)
    return layer, torch.randn(1, SEQ, HIDDEN, device=DEV) * 0.1


def test_input_rms_norm_evaluate():
    """input_rms_norm vs HF `input_layernorm`. `Gemma2RMSNorm` scales by
    `1.0 + weight`, so `gamma_in` comes through `config.rms_gamma`."""
    layer, x = _one_token()

    with torch.no_grad():
        ref = layer.input_layernorm(x)
    out = evaluate(
        model.input_rms_norm, x, config.rms_gamma(layer.input_layernorm), device=DEV
    )

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_mlp_evaluate():
    """mlp (the pure dense `gelu_pytorch_tanh`-gated block, no norm on either
    side) vs HF `Gemma2MLP`, over the input `pre_feedforward_layernorm` hands it.

    Named so `pytest -k gelu` selects it: `gelu_pytorch_tanh` is what this model
    has where the Qwen siblings have SwiGLU's `silu`.
    """
    layer, x = _one_token()
    mlp = layer.mlp

    with torch.no_grad():
        normed = layer.pre_feedforward_layernorm(x)
        ref = mlp(normed)

    out = evaluate(
        model.mlp,
        normed,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_self_attention_evaluate():
    """self_attention (GQA + RoPE + `query_pre_attn_scalar` scaling +
    `attn_logit_softcapping`, over the cache and the new token) vs HF's own
    attention at the decoded position.

    At the drawn scale the raw logits reach 18.8 against a cap of 50, so `tanh`
    has already bent: dropping the *cache* group's cap moves the output by 0.218,
    55x the tolerance here. It does not hold the *new token's own* cap to account,
    though -- that is one logit of twenty-five, and dropping its cap moves the
    output by 5.3e-05, 0.014x the tolerance. Both caps are pinned by
    `test_self_attention_soft_caps_the_new_token_too`.
    """
    drawn = reference.decode_step_inputs(device=DEV)
    fn = specialize_concretely(model.self_attention, {"ctx_len": drawn.ctx_len})
    out, _, _ = evaluate(fn, *drawn.attention_args, device=DEV)

    cfg = config.build_hf_config()
    total = drawn.ctx_len + SEQ
    cos, sin = config.rope_caches(cfg, total, device=DEV)
    mask = oracle.causal_mask(total, DEV)
    sequence = torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1)
    with torch.no_grad():
        normed = drawn.layer.input_layernorm(sequence)
        ref, _ = drawn.layer.self_attn(
            normed,
            position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=mask,
        )

    torch.testing.assert_close(
        out.float(), ref[:, -SEQ:, :].float(), atol=ATOL, rtol=RTOL
    )


def test_self_attention_soft_caps_the_new_token_too():
    """The soft cap is on the new token's own logit as well as the cache's.

    A step that capped only the cache group would pass every other test in this
    file. The new token contributes one logit of `ctx_len + 1`, and at the drawn
    scale leaving it uncapped moves the output by 0.014x the tolerance -- below
    the round-off the comparison already allows. So the query is scaled until that
    single logit is far past the cap, where the same omission moves the output by
    9329x the tolerance instead, and the kernel is asked again.

    `scale` is the tenth of `self_attention`'s eleven arguments; x100 puts the raw
    logits at ~1900 against a cap of 50.
    """
    drawn = reference.decode_step_inputs(device=DEV)
    fn = specialize_concretely(model.self_attention, {"ctx_len": drawn.ctx_len})

    loud = list(drawn.attention_args)
    loud[9] = loud[9] * SOFTCAP_PROBE_SCALE
    out, _, _ = evaluate(fn, *loud, device=DEV)

    cfg = config.build_hf_config()
    total = drawn.ctx_len + SEQ
    cos, sin = config.rope_caches(cfg, total, device=DEV)
    mask = oracle.causal_mask(total, DEV)
    sequence = torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1)
    attention = drawn.layer.self_attn
    with torch.no_grad():
        normed = drawn.layer.input_layernorm(sequence)
        held = attention.scaling
        attention.scaling = held * SOFTCAP_PROBE_SCALE
        try:
            ref, _ = attention(
                normed,
                position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
                attention_mask=mask,
            )
        finally:
            attention.scaling = held

    torch.testing.assert_close(
        out.float(), ref[:, -SEQ:, :].float(), atol=ATOL, rtol=RTOL
    )


def test_decoder_layer_evaluate():
    """Full decoder_layer -- `h = x + post_attn_norm(attn(input_norm(x)))`, then
    `out = h + post_ff_norm(mlp(pre_ff_norm(h)))` -- vs the complete HF
    `Gemma2DecoderLayer.forward` at the decoded position, at two context
    lengths."""
    for ctx_len in CTX_LENGTHS:
        drawn = reference.decode_step_inputs(ctx_len=ctx_len, device=DEV)
        fn = specialize_concretely(model.decoder_layer, {"ctx_len": ctx_len})
        out, _, _ = evaluate(fn, *drawn.args, device=DEV)

        want = reference.decode_step_oracle(drawn)
        torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_decoder_layer_returns_the_cache_entry_to_append():
    """The step's returned key and value are this token's cache entry: appending
    them to the cache it was given reproduces the cache a context one token
    longer would have produced.

    Checked against a rebuilt cache rather than against the step's own inputs,
    so a step that returned its inputs unchanged would fail.
    """
    drawn = reference.decode_step_inputs(device=DEV)
    fn = specialize_concretely(model.decoder_layer, {"ctx_len": drawn.ctx_len})
    _, k_new, v_new = evaluate(fn, *drawn.args, device=DEV)

    want_k, want_v = reference.appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert tuple(grown_k.shape) == tuple(want_k.shape)
    torch.testing.assert_close(grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL)
