"""Gemma-2-2B, as the installation ships it, asked through the commands."""

from __future__ import annotations

import torch

from tests.core.models.decode_oracle import SEQ_LEN, causal_mask
from tests.nvidia.cuda.installed.models import contract
from tests.nvidia.cuda.models.gemma2_2b import reference

MODEL = "gemma2_2b"
CASES = contract.model_cases(MODEL)


def test_decode_step_returns_cache_entry(tf, shipped_source, comparison_cache) -> None:
    """One decode step of one layer, and the state the step hands back.

    The whole layer is compared with ``Gemma2DecoderLayer.forward``. Returned cache
    entries are checked against oracle state rebuilt one token longer, so returning
    inputs unchanged fails. The supplied cache is oracle state, making the appended
    entry the only computed part. Every returned output is judged.
    """
    drawn = reference.decode_step_inputs(device="cpu")
    source, case = shipped_source(MODEL), CASES[0]
    want_out = reference.decode_step_oracle(drawn)
    want_k, want_v = reference.appended_cache_oracle(drawn)
    entry_k, entry_v = want_k[:, drawn.ctx_len :], want_v[:, drawn.ctx_len :]

    contract.compared(
        tf,
        comparison_cache,
        source,
        case,
        "decoder_layer",
        activations=drawn.args,
        weights={name: drawn.loaded.resource.load(name) for name in drawn.loaded.module.weights},
        expected=(want_out, entry_k, entry_v),
        held=(
            contract.three_roundings(want_out),
            contract.three_roundings(entry_k),
            contract.three_roundings(entry_v),
        ),
        dims={"ctx_len": drawn.ctx_len},
    )

    assert want_k.shape[1] == drawn.ctx_len + 1
    assert entry_k.shape[1] == 1 and entry_v.shape[1] == 1


def test_attention_matches_hugging_face(tf, shipped_source, comparison_cache) -> None:
    """The attention matches Hugging Face at the decoded position.

    `self_attention` -- input_layernorm plus Gemma2's GQA, RoPE and soft-capped
    logits over the cache and the new token -- against Hugging Face's own attention
    at the decoded position.

    At the ordinary decode input scale, which is the scale the comparison means
    something at: the soft cap is part of the computation being reproduced, not a
    thing to be provoked, and driving the query past the cap only compresses the
    reference range until the bound measures the compression.
    """
    drawn = reference.decode_step_inputs(device="cpu")
    source, case = shipped_source(MODEL), CASES[0]

    total = drawn.ctx_len + SEQ_LEN
    cos, sin = reference._rope_at(total, "cpu")
    sequence = torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1)

    mask = causal_mask(total, "cpu", sequence.dtype)
    with torch.no_grad():
        reference_out, _ = drawn.layer.self_attn(
            drawn.layer.input_layernorm(sequence),
            position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=mask,
        )
    want = reference_out[:, -SEQ_LEN:, :]
    want_k, want_v = reference.appended_cache_oracle(drawn)
    entry_k, entry_v = want_k[:, drawn.ctx_len :], want_v[:, drawn.ctx_len :]

    contract.compared(
        tf,
        comparison_cache,
        source,
        case,
        "self_attention",
        activations=drawn.attention_args,
        weights={name: drawn.loaded.resource.load(name) for name in drawn.loaded.module.weights},
        expected=(want, entry_k, entry_v),
        held=(
            contract.three_roundings(want),
            contract.one_rounding(entry_k),
            contract.one_rounding(entry_v),
        ),
        dims={"ctx_len": drawn.ctx_len},
    )


test_model_analysis, test_model_ceiling = contract.analysis_tests(MODEL)
