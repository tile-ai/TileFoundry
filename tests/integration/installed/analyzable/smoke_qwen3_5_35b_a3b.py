"""Qwen3.5-35B-A3B, as the installation ships it, asked through the commands.

One root reached three ways: the linear-attention mixer, the full-attention mixer of
a later layer, and the MoE block. They are three corpus entries because a Module is
the execution domain of the functions it owns, and only the full-attention one leaves
a context length open to be asked at a size.
"""
from __future__ import annotations

import json

import contract
import pytest
import torch

from tests.models.qwen3_5_35b_a3b import reference

MODEL = "qwen3_5_35b_a3b"
CASES = contract.model_cases(MODEL)

ANALYSED = [
    pytest.param(case, selected, family, id=f"{selected.id}/{family}")
    for case in CASES
    for selected in case.analyze
    for family in contract.FAMILIES
]
PLANNED = [
    pytest.param(case, planned, id=planned.id)
    for case in CASES
    for planned in case.schedule
]
#: One case per Module, as the levels a root declares are a property of the root.
FIRST_PLAN = [pytest.param(case, case.schedule[0], id=case.id) for case in CASES]
SIZED = [pytest.param(case, sized, id=sized.id) for case in CASES for sized in case.sized]

#: The bindings whose cost is the context, per case: only the full-attention mixer
#: has one to zero.
ZERO_SIZED = {
    "qwen3_5_35b_a3b_full_attention": frozenset(
        ("k_cache", "v_cache", "k_ctx", "score_ctx", "p_ctx", "weighted")
    ),
}


@pytest.mark.parametrize(("case", "selected", "family"), ANALYSED)
def test_every_selected_function_analyses(tf, shipped_source, case, selected, family) -> None:
    contract.analysed(
        tf, shipped_source(MODEL), case, selected.selector, family, selected.dims
    )


@pytest.mark.parametrize(("case", "planned"), PLANNED)
def test_every_selected_function_plans(tf, shipped_source, case, planned) -> None:
    contract.scheduled(tf, shipped_source(MODEL), case, planned)


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_each_model_is_asked_at_a_size(tf, shipped_source, case, sized) -> None:
    contract.analysed(
        tf, shipped_source(MODEL), case, sized.selector, "compute-cost", sized.dims
    )


@pytest.mark.parametrize(("case", "sized"), SIZED)
@pytest.mark.parametrize("family", contract.FAMILIES)
def test_every_analysis_answers_at_the_largest_context(
    tf, shipped_source, case, sized, family
) -> None:
    """At the ceiling the case states, not at a sample of it."""
    contract.analysed(
        tf, shipped_source(MODEL), case, sized.selector, family, sized.ceiling
    )


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_the_ceiling_is_reasoned_about_at_its_stated_length(
    tf, shipped_source, case, sized
) -> None:
    """What the analysis reports has to grow with the context.

    Growth rather than an absolute number: an analysis quietly working at a length
    it could afford instead of the one it was asked about would report the same
    footprint at both, and a number nobody compares would not show it.
    """
    source = shipped_source(MODEL)
    short = contract.traffic_read(tf, source, case, sized.selector, sized.dims)
    full = contract.traffic_read(tf, source, case, sized.selector, sized.ceiling)

    assert full > short, (
        f"analysing at {dict(sized.ceiling)} reports no more traffic than at "
        f"{dict(sized.dims)}, so the stated length changed nothing"
    )


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_the_open_dimensions_are_analysed_at_zero(tf, shipped_source, case, sized) -> None:
    """A binding whose whole cost is the context has to cost nothing without one."""
    source = shipped_source(MODEL)
    bindings = ZERO_SIZED[case.id]
    zero = contract.lifetimes(
        tf, source, case, sized.selector, {name: 0 for name in sized.dims}
    )
    nonzero = contract.lifetimes(tf, source, case, sized.selector, sized.dims)

    assert bindings <= zero.keys()
    assert all(zero[binding] == 0 for binding in bindings)
    assert all(nonzero[binding] > 0 for binding in bindings)


@pytest.mark.parametrize(("case", "planned"), FIRST_PLAN)
def test_the_command_reports_a_real_model_as_json(tf, shipped_source, case, planned) -> None:
    done = contract.analysed(
        tf,
        shipped_source(MODEL),
        case,
        planned.selector,
        "compute-cost",
        planned.dims,
        json_output=True,
    )

    assert json.loads(done.stdout)


@pytest.mark.parametrize(("case", "planned"), FIRST_PLAN)
def test_the_command_reads_the_machine_off_the_shipped_source(
    tf, shipped_source, case, planned
) -> None:
    """Nothing tells the command which target to use; the source has to say."""
    done = contract.capabilities(tf, shipped_source(MODEL), case, planned.selector)

    assert done.stdout.strip()


# ── against Hugging Face ─────────────────────────────────────────────────────
#: Two lengths, so a mixer that only works at the length it was authored against
#: cannot pass.
CTX_LENGTHS = (25, 40)

FULL = next(case for case in CASES if case.id.endswith("full_attention"))
LINEAR = next(case for case in CASES if case.id == "qwen3_5_35b_a3b")


@pytest.mark.parametrize("ctx_len", CTX_LENGTHS)
def test_full_attention_matches_hugging_face(tf, shipped_source, tmp_path, ctx_len) -> None:
    """`full_attention` -- input_layernorm plus GQA with per-head q_norm/k_norm,
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
        tf, tmp_path, shipped_source(MODEL), FULL, "full_attention",
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
    """`linear_attention` -- input_layernorm plus the causal convolution,
    L2-normalised query and key, the gated delta rule and the gated output norm --
    against Hugging Face's own mixer at the decoded position, at two lengths.

    The returned convolution column and recurrent matrix are the state a caller
    holds afterwards, compared against a state rebuilt one token longer. For the
    recurrent matrix that is the failure worth guarding, since its shape gives
    nothing away. No `--dim`: this mixer's state is a fixed-size recurrent matrix
    rather than a growing cache, so it leaves no dimension open to bind -- the two
    lengths differ in the context the oracle was drawn over, not in the kernel.
    """
    step = reference.linear_step(ctx_len=ctx_len, device="cpu")
    loaded = reference.load_mixer("linear_attention", step.layer)
    want = reference.linear_mixer_oracle(step)
    want_conv, want_state = reference.advanced_state_oracle(step)
    # The window slides on its last axis, so this token's column is the newest one.
    entry = want_conv[..., -1:]

    contract.compared(
        tf, tmp_path, shipped_source(MODEL), LINEAR, "linear_attention",
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
    """The whole block -- post_attention_layernorm plus `Qwen3_5MoeSparseMoeBlock`,
    routed experts and the shared expert together -- against Hugging Face's own.

    Named as a Module so `check` compares the block's own orchestration rather than
    one of its functions: the routed and shared halves are summed inside it.
    """
    step = reference.linear_step(device="cpu", whole_layer=True)
    loaded = reference.load_moe(step.layer)
    want = reference.moe_oracle(step.layer, step.hidden_new)

    contract.compared(
        tf, tmp_path, shipped_source(MODEL), MOE, "",
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
        tf, work, source, LINEAR, "linear_attention",
        activations=activations,
        weights=contract.nested_constants(loaded),
        expected=(want, entry, want_state),
        held=(
            contract.three_roundings(want),
            contract.three_roundings(entry),
            contract.three_roundings(want_state),
        ),
    )


def test_the_prior_state_is_read(tf, shipped_source, tmp_path) -> None:
    """The recurrent matrix handed in reaches the answer.

    A linear-attention step has no `ctx_len` in its signature, so nothing about its
    shape says it consulted the context at all -- an implementation that dropped the
    incoming matrix would produce a plausible tensor of the right size. Measured by
    zeroing the matrix: that has to move the answer away from the oracle the
    unperturbed step meets, or the kernel is not reading it and every agreement above
    would be an agreement about one token in isolation.
    """
    step = reference.linear_step(device="cpu")
    loaded = reference.load_mixer("linear_attention", step.layer)

    _linear_disagrees(
        tf, tmp_path, shipped_source(MODEL), step, loaded,
        (step.hidden_new, step.conv_state, torch.zeros_like(step.recurrent_state)),
    )


def test_the_convolution_window_is_read(tf, shipped_source, tmp_path) -> None:
    """The convolution's left context reaches the answer.

    The same argument as the recurrent matrix, for the other half of the state: at
    one token per step, a kernel that convolved only the current column would be a
    kernel with a kernel size of one, and nothing about its output shape would say so.
    """
    step = reference.linear_step(device="cpu")
    loaded = reference.load_mixer("linear_attention", step.layer)

    _linear_disagrees(
        tf, tmp_path, shipped_source(MODEL), step, loaded,
        (step.hidden_new, torch.zeros_like(step.conv_state), step.recurrent_state),
    )


def test_the_output_gate_is_applied(tf, shipped_source, tmp_path) -> None:
    """Half of `q_proj`'s fan-out never reaches a score, and this measures that it
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

    # w_qg is [1, hidden, 2 * heads * head_dim] with the gate interleaved per head.
    neutral = dict(loaded.constants)
    gated = neutral["w_qg"].clone().reshape(
        1, shape.hidden_size, shape.num_attention_heads, 2 * shape.head_dim
    )
    gated[..., shape.head_dim:] = 0.0
    neutral["w_qg"] = gated.reshape(loaded.constants["w_qg"].shape)

    contract.disagreed(
        tf, tmp_path, shipped_source(MODEL), FULL, "full_attention",
        activations=(step.hidden_new, *step.mixer_acts),
        weights=neutral,
        expected=(want, want_key[:, step.ctx_len:], want_value[:, step.ctx_len:]),
        held=(
            contract.three_roundings(want),
            contract.three_roundings(want_key[:, step.ctx_len:]),
            contract.three_roundings(want_value[:, step.ctx_len:]),
        ),
        dims={"ctx_len": step.ctx_len},
    )
