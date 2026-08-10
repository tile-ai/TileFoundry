"""Qwen3.5-35B-A3B, as the installation ships it, asked through the commands.

One root reached three ways: the linear-attention mixer, the full-attention mixer of
a later layer, and the MoE block. They are three corpus entries because a Module is
the execution domain of the functions it owns, and only the full-attention one leaves
a context length open to be asked at a size.
"""

from __future__ import annotations

import contract
import pytest
import torch

from tests.models.qwen3_5_35b_a3b import reference

MODEL = "qwen3_5_35b_a3b"
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


@pytest.mark.parametrize(("case", "planned"), PLANNED)
def test_every_selected_function_plans(tf, shipped_source, case, planned) -> None:
    contract.scheduled(tf, shipped_source(MODEL), case, planned)


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_every_analysis_answers_at_the_largest_context(tf, shipped_source, case, sized) -> None:
    """At the ceiling the case states, not at a sample of it."""
    contract.analysed_every_family(tf, shipped_source(MODEL), case, sized.selector, sized.ceiling)


CTX_LENGTHS = (25, 40)

FULL = next(case for case in CASES if case.id.endswith("full_attention"))
LINEAR = next(case for case in CASES if case.id == "qwen3_5_35b_a3b")


@pytest.mark.parametrize("ctx_len", CTX_LENGTHS)
def test_full_attention_matches_hugging_face(tf, shipped_source, tmp_path, ctx_len) -> None:
    """Test full attention matches hugging face.

    `full_attention` -- input_layernorm plus GQA with per-head q_norm/k_norm,
    partial RoPE and the output gate, over the cache and the new token -- against
    Hugging Face's own attention at the decoded position, at two lengths.

    The returned key and value are this token's cache entry, compared against a
    cache rebuilt one token longer rather than against the step's own inputs. The
    cache each entry is appended to is the oracle's own, so this token's entry is
    the only computed part and the one the bound follows.
    """
    step = reference.full_step(ctx_len=ctx_len, device="cpu")
    loaded = reference.load_mixer("full_attention", step.layer)
    want = reference.full_mixer_oracle(step)
    want_key, want_value = reference.appended_cache_oracle(step)
    entry_key, entry_value = want_key[:, ctx_len:], want_value[:, ctx_len:]

    assert want_key.shape[1] == ctx_len + 1
    contract.compared(
        tf,
        tmp_path,
        shipped_source(MODEL),
        FULL,
        "full_attention",
        activations=(step.hidden_new, *step.mixer_acts),
        weights=loaded.constants,
        expected=(want, entry_key, entry_value),
        held=(
            contract.three_roundings(want),
            contract.three_roundings(entry_key),
            contract.three_roundings(entry_value),
        ),
        dims={"ctx_len": ctx_len},
    )


@pytest.mark.parametrize("ctx_len", CTX_LENGTHS)
def test_linear_attention_matches_hugging_face(tf, shipped_source, tmp_path, ctx_len) -> None:
    """Test linear attention matches hugging face.

    Compare normalization, convolution, gated delta rule, and output normalization
    with Hugging Face at two lengths. Returned convolution and recurrent state are
    checked against an oracle advanced one token. The fixed-size state exposes no
    ``--dim``; lengths vary oracle context, not kernel shape.
    """
    step = reference.linear_step(ctx_len=ctx_len, device="cpu")
    loaded = reference.load_mixer("linear_attention", step.layer)
    want = reference.linear_mixer_oracle(step)
    want_conv, want_state = reference.advanced_state_oracle(step)

    entry = want_conv[..., -1:]

    contract.compared(
        tf,
        tmp_path,
        shipped_source(MODEL),
        LINEAR,
        "linear_attention",
        activations=(step.hidden_new, *step.mixer_acts),
        weights=loaded.constants,
        expected=(want, entry, want_state),
        held=(
            contract.three_roundings(want),
            contract.three_roundings(entry),
            contract.three_roundings(want_state),
        ),
    )


MOE = next(case for case in CASES if case.id.endswith("_moe"))


def test_the_moe_block_matches_hugging_face(tf, shipped_source, tmp_path) -> None:
    """Test the moe block matches hugging face.

    The whole block -- post_attention_layernorm plus `Qwen3_5MoeSparseMoeBlock`,
    routed experts and the shared expert together -- against Hugging Face's own.

    Named as a Module so `check` compares the block's own orchestration rather than
    one of its functions: the routed and shared halves are summed inside it.
    """
    step = reference.linear_step(device="cpu", whole_layer=True)
    loaded = reference.load_moe(step.layer)
    want = reference.moe_oracle(step.layer, step.hidden_new)

    contract.compared(
        tf,
        tmp_path,
        shipped_source(MODEL),
        MOE,
        "",
        activations=(step.hidden_new,),
        weights=contract.nested_constants(loaded),
        expected=(want,),
        held=(contract.three_roundings(want),),
    )


def _linear_disagrees(tf, work, source, step, loaded, activations) -> None:
    """A perturbed linear step has to move away from the oracle it otherwise meets."""
    want = reference.linear_mixer_oracle(step)
    want_conv, want_state = reference.advanced_state_oracle(step)
    entry = want_conv[..., -1:]

    contract.disagreed(
        tf,
        work,
        source,
        LINEAR,
        "linear_attention",
        activations=activations,
        weights=contract.nested_constants(loaded),
        expected=(want, entry, want_state),
        held=(
            contract.three_roundings(want),
            contract.three_roundings(entry),
            contract.three_roundings(want_state),
        ),
    )


ZEROED = ["recurrent_state", "conv_state"]


@pytest.mark.parametrize("zeroed", ZEROED)
def test_each_half_of_the_state_reaches_the_answer(tf, shipped_source, tmp_path, zeroed) -> None:
    step = reference.linear_step(device="cpu")
    loaded = reference.load_mixer("linear_attention", step.layer)
    held = {"conv_state": step.conv_state, "recurrent_state": step.recurrent_state}
    held[zeroed] = torch.zeros_like(held[zeroed])

    _linear_disagrees(
        tf,
        tmp_path,
        shipped_source(MODEL),
        step,
        loaded,
        (step.hidden_new, held["conv_state"], held["recurrent_state"]),
    )


def test_the_output_gate_is_applied(tf, shipped_source, tmp_path) -> None:
    """Half of `q_proj`'s fan-out never reaches a score.

    Half of `q_proj`'s fan-out never reaches a score, and this measures that it
    reaches the output instead.

    The gate is a sigmoid, so it lies strictly between 0 and 1: an implementation
    that ignored it would be uniformly larger, and one that applied it twice
    uniformly smaller. Both are caught by running the same step against a checkpoint
    whose gate half is zeroed -- every gate becomes sigmoid(0) = 1/2, so if the
    answer did not move, the gate is not being read.
    """
    step = reference.full_step(device="cpu")
    loaded = reference.load_mixer("full_attention", step.layer)
    shape = reference.CONFIG
    want = reference.full_mixer_oracle(step)
    want_key, want_value = reference.appended_cache_oracle(step)

    neutral = dict(loaded.constants)
    gated = (
        neutral["w_qg"]
        .clone()
        .reshape(1, shape.hidden_size, shape.num_attention_heads, 2 * shape.head_dim)
    )
    gated[..., shape.head_dim :] = 0.0
    neutral["w_qg"] = gated.reshape(loaded.constants["w_qg"].shape)

    contract.disagreed(
        tf,
        tmp_path,
        shipped_source(MODEL),
        FULL,
        "full_attention",
        activations=(step.hidden_new, *step.mixer_acts),
        weights=neutral,
        expected=(want, want_key[:, step.ctx_len :], want_value[:, step.ctx_len :]),
        held=(
            contract.three_roundings(want),
            contract.three_roundings(want_key[:, step.ctx_len :]),
            contract.three_roundings(want_value[:, step.ctx_len :]),
        ),
        dims={"ctx_len": step.ctx_len},
    )
