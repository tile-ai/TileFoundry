"""Gemma-2-2B, as the installation ships it, asked through the commands."""
from __future__ import annotations

import contract
import pytest
import torch

from tests.models.decode_oracle import SEQ_LEN, causal_mask
from tests.models.gemma2_2b import reference

MODEL = "gemma2_2b"
CASES = contract.model_cases(MODEL)

ANALYSED = [
    pytest.param(case, selected, id=selected.id)
    for case in CASES
    for selected in case.analyze
]
PLANNED = [
    pytest.param(case, planned, id=planned.id)
    for case in CASES
    for planned in case.schedule
]
SIZED = [pytest.param(case, sized, id=sized.id) for case in CASES for sized in case.sized]

@pytest.mark.parametrize(("case", "selected"), ANALYSED)
def test_every_selected_function_analyses(tf, shipped_source, case, selected) -> None:
    contract.analysed_every_family(
        tf, shipped_source(MODEL), case, selected.selector, selected.dims
    )


@pytest.mark.parametrize(("case", "planned"), PLANNED)
def test_every_selected_function_plans(tf, shipped_source, case, planned) -> None:
    contract.scheduled(tf, shipped_source(MODEL), case, planned)


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_every_analysis_answers_at_the_largest_context(
    tf, shipped_source, case, sized
) -> None:
    """At the ceiling the case states, not at a sample of it."""
    contract.analysed_every_family(
        tf, shipped_source(MODEL), case, sized.selector, sized.ceiling
    )


# ── against Hugging Face ─────────────────────────────────────────────────────
def test_the_decode_step_and_the_cache_entry_it_hands_back(
    tf, shipped_source, tmp_path
) -> None:
    """One decode step of one layer, and the state the step hands back.

    The boundary: the whole layer -- the norms, GQA attention with RoPE and the
    soft-capped logits over the cache and the new token, the residual and the MLP --
    against `Gemma2DecoderLayer.forward`.

    The returned key and value are this token's cache entry: they are compared
    against a cache rebuilt over a context one token longer, not against the step's
    own inputs, so a step that returned its inputs unchanged fails. The cache handed
    in is the oracle's own, so the appended entry is the only computed part and the
    one whose precision the bound follows.

    The command requires a predicate for every output a function returns, so the
    step's own output is judged here too rather than discarded.
    """
    drawn = reference.decode_step_inputs(device="cpu")
    source, case = shipped_source(MODEL), CASES[0]
    want_out = reference.decode_step_oracle(drawn)
    want_k, want_v = reference.appended_cache_oracle(drawn)
    entry_k, entry_v = want_k[:, drawn.ctx_len:], want_v[:, drawn.ctx_len:]

    contract.compared(
        tf, tmp_path, source, case, "decoder_layer",
        activations=drawn.args,
        weights=drawn.loaded.constants,
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


def test_the_attention_matches_hugging_face(tf, shipped_source, tmp_path) -> None:
    """`self_attention` -- input_layernorm plus Gemma2's GQA, RoPE and soft-capped
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
    # At the activations dtype: an f32 mask would promote HF attention and make this
    # a bf16-against-f32 comparison rather than a comparison of two kernels.
    mask = causal_mask(total, "cpu", sequence.dtype)
    with torch.no_grad():
        reference_out, _ = drawn.layer.self_attn(
            drawn.layer.input_layernorm(sequence),
            position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=mask,
        )
    want = reference_out[:, -SEQ_LEN:, :]
    want_k, want_v = reference.appended_cache_oracle(drawn)
    entry_k, entry_v = want_k[:, drawn.ctx_len:], want_v[:, drawn.ctx_len:]

    contract.compared(
        tf, tmp_path, source, case, "self_attention",
        activations=drawn.attention_args,
        weights=drawn.loaded.constants,
        expected=(want, entry_k, entry_v),
        held=(
            contract.three_roundings(want),
            contract.one_rounding(entry_k),
            contract.one_rounding(entry_v),
        ),
        dims={"ctx_len": drawn.ctx_len},
    )
