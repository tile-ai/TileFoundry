"""MiniCPM3-4B decode step: resolve a kernel by name, evaluate vs HF.

cpu + f32. Each test resolves one kernel from the ``MiniCPM3_4B`` module and
checks it against the corresponding Hugging Face ``MiniCPM3DecoderLayer``
submodule(s).

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

from tests.models import decode_oracle as oracle
from tests.models.minicpm3_4b import config, reference
from tests.models.minicpm3_4b import minicpm3_4b as model
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.specialize import specialize_concretely

HIDDEN = config.REAL.hidden
SEQ = config.SEQ_LEN

DEV = "cpu"
ATOL = RTOL = 2e-4

#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither is a multiple of the head count.
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
    out = evaluate(model.input_rms_norm, x, layer.input_layernorm.weight, device=DEV)

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=RTOL)


def test_mlp_evaluate():
    """mlp (post_attention_layernorm + dense SwiGLU) vs HF."""
    layer, x = _one_token()
    mlp = layer.mlp

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


def test_mla_attention_evaluate():
    """mla_attention (input_layernorm + MLA self_attn: low-rank Q, shared
    low-rank KV latent, rotary-slice-only RoPE, MQA-shared k_rope, online softmax
    over the cache and the token) vs HF's own attention at the decoded position.
    """
    drawn = reference.decode_step_inputs(device=DEV)
    fn = specialize_concretely(model.mla_attention, {"ctx_len": drawn.ctx_len})
    # mla_attention takes the layer's arguments up to w_o; the MLP's four and the
    # residual scale come after it in decoder_layer's signature.
    out, _, _ = evaluate(fn, *drawn.args[:-5], device=DEV)

    cfg = config.build_hf_config()
    total = drawn.ctx_len + SEQ
    cos, sin = config.rope_caches(cfg, total, device=DEV)
    sequence = torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1)
    with torch.no_grad():
        normed = drawn.layer.input_layernorm(sequence)
        ref, _ = drawn.layer.self_attn(
            normed,
            position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=oracle.causal_mask(total, DEV, sequence.dtype),
        )

    torch.testing.assert_close(
        out.float(), ref[:, -SEQ:, :].float(), atol=ATOL, rtol=RTOL
    )


def test_decoder_layer_evaluate():
    """Full decoder_layer (mla_attention + scale_depth residual + mlp +
    scale_depth residual) vs the complete HF `MiniCPM3DecoderLayer.forward` at
    the decoded position, at two context lengths."""
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

    For MLA that is the claim the whole cache design rests on -- the entry is the
    assembled per-head key and the up-projected value, which is what Hugging
    Face's own cache holds (see ``config.py``). Checked against a rebuilt cache
    rather than against the step's own inputs, so a step that returned its inputs
    unchanged would fail.
    """
    drawn = reference.decode_step_inputs(device=DEV)
    fn = specialize_concretely(model.decoder_layer, {"ctx_len": drawn.ctx_len})
    _, k_new, v_new = evaluate(fn, *drawn.args, device=DEV)

    want_k, want_v = reference.appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert tuple(grown_k.shape) == tuple(want_k.shape)
    assert tuple(grown_v.shape) == tuple(want_v.shape)
    torch.testing.assert_close(grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL)
