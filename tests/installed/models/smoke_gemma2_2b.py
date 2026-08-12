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
    pytest.param(case, selected, id=selected.id) for case in CASES for selected in case.analyze
]
PLANNED = [
    pytest.param(case, planned, id=planned.id) for case in CASES for planned in case.schedule
]
SIZED = [pytest.param(case, sized, id=sized.id) for case in CASES for sized in case.sized]


@pytest.mark.parametrize(("case", "selected"), ANALYSED)
def test_every_selected_function_analyses(tf, shipped_source, case, selected) -> None:
    contract.analysed_every_family(
        tf, shipped_source(MODEL), case, selected.selector, selected.dims
    )


def test_unplaced_model_refuses_timeline(tf, shipped_source) -> None:
    contract.timeline_refused(tf, shipped_source(MODEL), CASES[0], CASES[0].analyze[0])


@pytest.mark.parametrize(("case", "planned"), PLANNED)
def test_every_selected_function_plans(tf, shipped_source, case, planned) -> None:
    contract.scheduled(tf, shipped_source(MODEL), case, planned)


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_every_analysis_answers_at_the_largest_context(tf, shipped_source, case, sized) -> None:
    """At the ceiling the case states, not at a sample of it."""
    contract.analysed_every_family(tf, shipped_source(MODEL), case, sized.selector, sized.ceiling)


def test_the_decode_step_and_the_cache_entry_it_hands_back(tf, shipped_source, tmp_path) -> None:
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
        tmp_path,
        source,
        case,
        "decoder_layer",
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
    """Test the attention matches hugging face.

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
        tmp_path,
        source,
        case,
        "self_attention",
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
