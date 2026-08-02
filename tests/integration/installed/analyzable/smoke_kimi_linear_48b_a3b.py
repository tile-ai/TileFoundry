"""Kimi-Linear-48B-A3B, as the installation ships it, asked through the commands.

Three Modules of kernels reached from one root, so this model states three schedule
cases where most state one, and its selectors are dotted paths through them. It
ships no ``hf_alias.py``: it is not loaded from a raw published checkpoint.
"""
from __future__ import annotations

import json

import contract
import pytest
import torch

from tests.models.kimi_linear_48b_a3b import reference

MODEL = "kimi_linear_48b_a3b"
CONFIG = reference.CONFIG
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

#: The bindings whose cost is the context, so a zero context has to zero them.
ZERO_SIZED = frozenset(("k_cache", "v_cache", "score_ctx", "p_ctx", "weighted"))


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
    bindings = ZERO_SIZED
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
#: Two lengths, so a kernel that only works at the length it was authored against
#: cannot pass. Neither is a multiple of the 32 heads.
CTX_LENGTHS = (24, 40)

#: What a perturbed run has to move the output by before the parity runs above count
#: as discriminating. Far above the rounding they accept, so "it changed" cannot be
#: round-off.
DISCRIMINATION = 1e-3

MLA = "mla.mla_attention"


def _mla(tf, work, source, step, *, args=None, refuse=False):
    """One `check` of the MLA step, judging its output and its cache entry."""
    activations, weights = contract.split_by_declaration(
        CASES[0], MLA, args if args is not None else step.args
    )
    want = reference.mla_step_oracle(step)
    want_k, want_v = reference.mla_appended_cache_oracle(step)
    entry_k, entry_v = want_k[:, step.ctx_len:], want_v[:, step.ctx_len:]
    ask = contract.disagreed if refuse else contract.compared
    held = (
        ("allclose", {"atol": DISCRIMINATION, "rtol": 0.0})
        if refuse
        else contract.three_roundings(want),
        contract.three_roundings(entry_k),
        contract.three_roundings(entry_v),
    )
    return ask(
        tf, work, source, CASES[0], MLA,
        activations=activations,
        weights=weights,
        expected=(want, entry_k, entry_v),
        held=held,
        dims={"ctx_len": step.ctx_len},
    )


@pytest.mark.parametrize("ctx_len", CTX_LENGTHS)
def test_mla_nope_matches_hugging_face(tf, shipped_source, tmp_path, ctx_len) -> None:
    """Kimi's own MLA form: NoPE, at two context lengths.

    NoPE does not drop the 64 rotary dimensions -- it stops rotating them. They still
    enter the score and the `qk_head_dim = 192` scaling denominator, which is why the
    scaling test below can tell the difference.
    """
    step = reference.mla_step_inputs(ctx_len=ctx_len, device="cpu", nope=True)
    _mla(tf, tmp_path, shipped_source(MODEL), step)


@pytest.mark.parametrize("ctx_len", CTX_LENGTHS)
def test_mla_rope_matches_hugging_face(tf, shipped_source, tmp_path, ctx_len) -> None:
    """The same kernel with a real rotary, at two context lengths.

    Not a configuration Kimi ships -- it is NoPE -- but it exercises the rotary path
    of the same kernel, and its agreement is what shows `tf.rope` and the oracle share
    the rotate-half convention.
    """
    step = reference.mla_step_inputs(ctx_len=ctx_len, device="cpu", nope=False)
    _mla(tf, tmp_path, shipped_source(MODEL), step)


def test_mla_returns_the_cache_entry_to_append(tf, shipped_source, tmp_path) -> None:
    """The returned key and value are this token's cache entry.

    Checked against a cache rebuilt over the context with the token appended, not
    against the step's own inputs, so a step that echoed its inputs would fail. The
    cache handed in is the oracle's own, so the appended entry is the only computed
    part and the one whose precision the bound follows.
    """
    step = reference.mla_step_inputs(device="cpu", nope=True)
    want_k, want_v = reference.mla_appended_cache_oracle(step)

    assert want_k.shape[1] == step.ctx_len + 1
    assert want_v.shape[1] == step.ctx_len + 1
    _mla(tf, tmp_path, shipped_source(MODEL), step)


def test_mla_scaling_is_qk_head_dim_not_v_head_dim(tf, shipped_source, tmp_path) -> None:
    """`qk_head_dim ** -0.5`, and the plausible wrong guess is detectable.

    Nothing in the published config says which dimension the score is scaled by, and
    `v_head_dim ** -0.5` is the natural guess: 0.0883883 against the correct
    0.0721688, 22.5% apart. Substituting it has to break the comparison -- which is
    what stops the parity runs above from passing on a wrong constant.
    """
    step = reference.mla_step_inputs(device="cpu", nope=True)
    args = list(step.args)
    args[11] = torch.full(
        (1, 1, 1, 1), CONFIG.v_head_dim ** -0.5, dtype=reference.DTYPE
    )

    _mla(tf, tmp_path, shipped_source(MODEL), step, args=args, refuse=True)


def test_mla_cache_pairing_is_load_bearing(tf, shipped_source, tmp_path) -> None:
    """Permuting one side of the cache breaks the answer.

    Softmax attention over a cache is permutation-invariant if both sides are
    permuted together -- position is already baked into the stored key -- so a joint
    permutation would prove nothing. Permuting the keys *without* the values breaks
    the pairing between them, and that must show. A kernel that read the cache at
    fixed offsets, or ignored it, would fail here.
    """
    step = reference.mla_step_inputs(device="cpu", nope=True)
    source = shipped_source(MODEL)
    torch.manual_seed(0)
    perm = torch.randperm(step.ctx_len)

    keys = list(step.args)
    keys[9] = step.k_cache[:, perm]
    _mla(tf, tmp_path, source, step, args=keys, refuse=True)

    values = list(step.args)
    values[10] = step.v_cache[:, perm]
    _mla(tf, tmp_path, source, step, args=values, refuse=True)


@pytest.mark.parametrize("act_seed", reference.MOE_DRAWS)
def test_moe_matches_hugging_face_at_the_published_expert_count(
    tf, shipped_source, tmp_path, act_seed
) -> None:
    """The full 256-expert MoE, over four independent draws.

    Four draws rather than one batch of four tokens: the decode contract fixes the
    token count at the literal 1, so breadth over which experts get selected has to
    come from redrawing. The four draws select genuinely different expert sets.
    """
    hf_moe = reference.build_hf_moe(device="cpu")
    try:
        step = reference.moe_inputs(device="cpu", act_seed=act_seed, hf_moe=hf_moe)
        want = reference.moe_oracle(step)
        activations, weights = contract.split_by_declaration(CASES[0], "moe.moe", step.args)
        contract.compared(
            tf, tmp_path, shipped_source(MODEL), CASES[0], "moe.moe",
            activations=activations,
            weights=weights,
            expected=(want,),
            held=(contract.three_roundings(want),),
        )
    finally:
        del hf_moe

