"""Kimi-Linear-48B-A3B, as the installation ships it, asked through the commands.

Three Modules of kernels reached from one root, so this model states three schedule
cases where most state one, and its selectors are dotted paths through them. It
ships no ``hf_alias.py``: it is not loaded from a raw published checkpoint.
"""
from __future__ import annotations

import contract
import pytest
import torch

from tests.models.kimi_linear_48b_a3b import reference

MODEL = "kimi_linear_48b_a3b"
CONFIG = reference.CONFIG
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
#: Two lengths, so a kernel that only works at the length it was authored against
#: cannot pass. Neither is a multiple of the 32 heads.
CTX_LENGTHS = (24, 40)

MLA = "mla.mla_attention"


def _mla(tf, work, source, step, *, args=None, refuse=False):
    """One `check` of the MLA step, judging its output and its cache entry.

    A perturbed run is held to breaking exactly the comparison the unperturbed run
    passes, so the two modes differ only in the verdict expected, never in the bound.
    A bound tighter than parity would be met by an unperturbed run as well, and the
    perturbation tests would pass without perturbing anything: measured, a fixed 1e-3
    refuses this step's output whose own parity bound is 0.00586, and the cache-pairing
    test passed under an identity permutation.
    """
    activations, weights = contract.split_by_declaration(
        CASES[0], MLA, args if args is not None else step.args
    )
    want = reference.mla_step_oracle(step)
    want_k, want_v = reference.mla_appended_cache_oracle(step)
    assert want_k.shape[1] == step.ctx_len + 1
    assert want_v.shape[1] == step.ctx_len + 1
    entry_k, entry_v = want_k[:, step.ctx_len:], want_v[:, step.ctx_len:]
    ask = contract.disagreed if refuse else contract.compared
    return ask(
        tf, work, source, CASES[0], MLA,
        activations=activations,
        weights=weights,
        expected=(want, entry_k, entry_v),
        held=(
            contract.three_roundings(want),
            contract.three_roundings(entry_k),
            contract.three_roundings(entry_v),
        ),
        dims={"ctx_len": step.ctx_len},
    )


#: Kimi's own MLA form is NoPE, which does not drop the 64 rotary dimensions -- it
#: stops rotating them. They still enter the score and the `qk_head_dim = 192` scaling
#: denominator, which is why the scaling test below can tell the difference. A real
#: rotary is not a configuration Kimi ships, but it exercises the rotary path of the
#: same kernel, and its agreement is what shows `tf.rope` and the oracle share the
#: rotate-half convention.
ROTATED = [
    *[pytest.param(ctx_len, True, id=f"nope/{ctx_len}") for ctx_len in CTX_LENGTHS],
    *[pytest.param(ctx_len, False, id=f"rope/{ctx_len}") for ctx_len in CTX_LENGTHS],
]


@pytest.mark.parametrize(("ctx_len", "nope"), ROTATED)
def test_mla_matches_hugging_face(tf, shipped_source, tmp_path, ctx_len, nope) -> None:
    step = reference.mla_step_inputs(ctx_len=ctx_len, device="cpu", nope=nope)

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


MOE = "moe.moe"


def _moe(tf, work, source, step, want, *, args=None, refuse=False):
    """One `check` of the MoE block against the oracle for the token it was drawn for.

    A perturbed run is held to breaking *this* comparison, for the reason `_mla` gives:
    this block's parity bound is `3 * one_ulp_at(want)` = 0.0234 at an oracle whose
    largest entry is 1.70, so a tighter fixed bound would refuse an unperturbed run too.
    """
    activations, weights = contract.split_by_declaration(
        CASES[0], MOE, args if args is not None else step.args
    )
    ask = contract.disagreed if refuse else contract.compared
    return ask(
        tf, work, source, CASES[0], MOE,
        activations=activations,
        weights=weights,
        expected=(want,),
        held=(contract.three_roundings(want),),
    )


def test_moe_matches_hugging_face(tf, shipped_source, tmp_path) -> None:
    """The full 256-expert MoE: four draws that agree, then three that must not.

    Four draws rather than one batch of four tokens, because the decode contract fixes
    the token count at the literal 1, so breadth over which experts get selected comes
    from redrawing. The three perturbations are asked at the published expert count by
    perturbing an input: every parameter `moe` declares is non-const, so each arrives
    as `--input`.

    One test, because the 256-expert block is the expensive part and all seven
    questions share it. They are asked in sequence, so the first to disagree -- or to
    agree when it should not -- fails at its own line.
    """
    hf_moe = reference.build_hf_moe()
    source = shipped_source(MODEL)
    drawn = {}
    for act_seed in reference.MOE_DRAWS:
        step = reference.moe_inputs(act_seed=act_seed, hf_moe=hf_moe)
        drawn[act_seed] = (step, reference.moe_oracle(step))
        _moe(tf, tmp_path, source, step, drawn[act_seed][1])

    step, want = drawn[reference.ACTIVATION_SEED]

    # The router selects on `sigmoid(logits) + bias` but takes the routing weights
    # from the *unbiased* scores. At bias = 0 those are the same tensor, so a
    # kernel that gathered the biased scores would be indistinguishable from a
    # correct one -- which is why the fixture draws the bias nonzero, and why that
    # must not be "simplified" to the zeros the class defaults to.
    biasless = list(step.args)
    biasless[3] = torch.zeros_like(step.args[3])
    _moe(tf, tmp_path, source, step, want, args=biasless, refuse=True)

    # `moe_renormalize: true` means normalise, *then* scale. The two orders are
    # different functions: scaling the selected scores before dividing by their sum
    # cancels the factor entirely, leaving what a factor of 1.0 would give. So a
    # kernel that folded the factor into the denominator has to fail here.
    unscaled = list(step.args)
    unscaled[4] = torch.full_like(step.args[4], 1.0)
    _moe(tf, tmp_path, source, step, want, args=unscaled, refuse=True)

    # `num_shared_experts: 1`, and it is unscaled -- `routed_scaling_factor`
    # applies to the routed branch only. Zeroing its gate projection drops the
    # shared branch's contribution, and a kernel that never added it could not pass
    # the parity runs above.
    unshared = list(step.args)
    unshared[8] = torch.zeros_like(step.args[8])
    _moe(tf, tmp_path, source, step, want, args=unshared, refuse=True)
