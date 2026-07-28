"""Qwen3-1.7B decode step: resolve a kernel by name, evaluate vs HF.

cpu + f32 oracle (no CUDA on this box — every ``device=`` below is ``"cpu"``).
Each test resolves one kernel from the ``Qwen3_1_7B`` module (mirroring
the convention every model package here shares) and checks it against the
corresponding Hugging Face ``Qwen3DecoderLayer`` submodule(s).

The kernels that read the KV cache carry ``ctx_len`` as a range, so they are
specialised at the length the drawn step uses before being evaluated: an extent
is what counting elements needs, and a range is not one. The kernels that do not
read the cache carry no range and are evaluated as authored.

Arguments come from ``reference.py``'s drawn step rather than being assembled
here, so the parameter order is stated once and a signature change cannot leave
one test agreeing with a stale order.
"""
from __future__ import annotations

import torch

from tests.models.qwen3_1_7b import config, reference
from tests.models.qwen3_1_7b import decoder_layer as qwen3
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.specialize import specialize_concretely

HIDDEN = config.REAL.hidden
SEQ = config.SEQ_LEN

DEV = "cpu"
ATOL = RTOL = 2e-4

#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither divides the key/value head count.
CTX_LENGTHS = (24, 40)


def _one_token(seed=1):
    """A fresh HF layer and one token's hidden states."""
    layer = config.build_hf_layer(seed=0, device=DEV)
    torch.manual_seed(seed)
    return layer, torch.randn(1, SEQ, HIDDEN, device=DEV) * 0.1


def test_input_rms_norm_evaluate():
    """input_rms_norm vs HF `input_layernorm`."""
    layer, x = _one_token()

    with torch.no_grad():
        ref = layer.input_layernorm(x)
    out = evaluate(qwen3.input_rms_norm, x, layer.input_layernorm.weight, device=DEV)

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_mlp_evaluate():
    """mlp (post_attention_layernorm + dense SwiGLU) vs HF."""
    layer, x = _one_token()
    mlp = layer.mlp

    with torch.no_grad():
        ref = mlp(layer.post_attention_layernorm(x))

    out = evaluate(
        qwen3.mlp,
        x,
        layer.post_attention_layernorm.weight,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
        device=DEV,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_tiled_mlp_matches_untiled_mlp():
    """tiled_mlp (the K-loop / column-block rewrite of `mlp`) against `mlp`
    itself on the same inputs: the loop tiling only reassociates the K
    reduction, so the two must agree to f32 round-off. Also checked against
    HF, so a bug shared by both rewrites cannot hide."""
    layer, x = _one_token()
    mlp = layer.mlp
    weights = (
        layer.post_attention_layernorm.weight,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
    )

    with torch.no_grad():
        ref = mlp(layer.post_attention_layernorm(x))
    untiled = evaluate(qwen3.mlp, x, *weights, device=DEV)
    tiled = evaluate(qwen3.tiled_mlp, x, *weights, device=DEV)

    torch.testing.assert_close(tiled.float(), untiled.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(tiled.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_self_attention_evaluate():
    """self_attention (input_layernorm + self_attn: GQA + RoPE + per-head
    q_norm/k_norm over the cache and the new token) vs HF's own attention at the
    decoded position."""
    drawn = reference.decode_step_inputs(device=DEV)
    fn = specialize_concretely(qwen3.self_attention, {"ctx_len": drawn.ctx_len})
    # self_attention takes the layer's arguments up to w_o; the MLP's four come
    # after it in decoder_layer's signature.
    out, _, _ = evaluate(fn, *drawn.args[:-4], device=DEV)

    cfg = config.build_hf_config()
    total = drawn.ctx_len + SEQ
    cos, sin = config.rope_caches(cfg, total, device=DEV)
    positions = torch.arange(total, device=DEV)
    mask = torch.where(
        positions.unsqueeze(0) <= positions.unsqueeze(1), 0.0, float("-inf")
    ).view(1, 1, total, total)
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


def test_decoder_layer_evaluate():
    """Full decoder_layer (self_attention + residual + mlp + residual) vs the
    complete HF `Qwen3DecoderLayer.forward` at the decoded position, at two
    context lengths."""
    for ctx_len in CTX_LENGTHS:
        drawn = reference.decode_step_inputs(ctx_len=ctx_len, device=DEV)
        fn = specialize_concretely(qwen3.decoder_layer, {"ctx_len": ctx_len})
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
    fn = specialize_concretely(qwen3.decoder_layer, {"ctx_len": drawn.ctx_len})
    _, k_new, v_new = evaluate(fn, *drawn.args, device=DEV)

    want_k, want_v = reference.appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert tuple(grown_k.shape) == tuple(want_k.shape)
    torch.testing.assert_close(grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL)
