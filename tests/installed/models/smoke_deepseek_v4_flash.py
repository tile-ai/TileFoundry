"""DeepSeek-V4-Flash, as the installation ships it, asked through the commands.

Its ``hf_alias.py`` sits in the checkout next to ``model.py`` and the packaging
manifest does not list it, so the directory this file asks about does not carry it.
Asking through the command is what makes that difference visible at all.
"""

from __future__ import annotations

import math

import contract
import pytest
import torch

from tests.models.deepseek_v4_flash import reference

MODEL = "deepseek_v4_flash"
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


def test_unplaced_model_refuses_performance(tf, shipped_source) -> None:
    contract.performance_refused(tf, shipped_source(MODEL), CASES[0], CASES[0].analyze[0])


@pytest.mark.parametrize(("case", "planned"), PLANNED)
def test_every_selected_function_plans(tf, shipped_source, case, planned) -> None:
    contract.scheduled(tf, shipped_source(MODEL), case, planned)


@pytest.mark.parametrize(("case", "sized"), SIZED)
def test_every_analysis_answers_at_the_largest_context(tf, shipped_source, case, sized) -> None:
    """At the ceiling the case states, not at a sample of it."""
    contract.analysed_every_family(tf, shipped_source(MODEL), case, sized.selector, sized.ceiling)


ATOL = RTOL = 2e-3


CTX_LENGTHS = (41, reference.CTX_LEN)


ATTENTION = ""

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernels this compares run on CUDA only"
)


def _bf16_ulps(want, count: int = 2) -> float:
    """*count* representable bf16 steps at *want*'s largest element.

    Derived rather than chosen: a bf16 significand is 8 bits, so within one binade
    the spacing is the binade divided by 128, and two answers that differ by one
    such step are the same answer written twice.
    """
    largest = want.float().abs().max().item()
    return count * 2.0 ** (math.floor(math.log2(largest)) - 7)


def _fp8_step(latent) -> float:
    """One e4m3 quantum of *latent*'s largest stored block.

    The cache holds its unrotated dims as fp8 with a per-block power-of-two scale,
    so two ways of computing the same latent may land one grid step apart. Derived
    from the block's own magnitude, so it cannot quietly widen.
    """
    blocks = latent[..., : reference.REAL.nope_dim].float().reshape(-1, reference.KV_QUANT_BLOCK)
    return blocks.abs().amax(dim=-1).max().item() / 8.0


def _entry(step):
    """The latent this token appends, and the grid it is stored on."""
    want = reference.appended_cache_oracle(step)
    return want, want[:, step.ctx_len :]


def _asked(tf, work, source, step, *, out_held):
    """One `check` of the attention step, judging both of its returns.

    No `--dim`: an orchestration method has no single signature to bind an extent
    against, so the context length is the one the supplied cache actually has.
    """
    _grown, entry = _entry(step)
    return contract.compared(
        tf,
        work,
        source,
        CASES[0],
        ATTENTION,
        activations=step.args,
        weights=step.weights,
        expected=(reference.attention_step_oracle(step), entry),
        held=(out_held, ("allclose", {"atol": _fp8_step(entry), "rtol": 0.0})),
    )


@cuda_only
def test_the_disagreement_is_smaller_than_the_oracles_own_rounding(
    tf, shipped_source, tmp_path
) -> None:
    """Test the disagreement is smaller than the oracles own rounding.

    Against an f32 accumulation of the same weights, the kernel is at least as
    close as Hugging Face's own bf16 run.

    Which is what says the comparison above is a comparison and not a tolerance wide
    enough to hide in. Stated as the oracle's own gap rather than a constant: the
    bound moves with the arithmetic, so a kernel that drifted inside a hand-picked
    tolerance cannot pass.
    """
    step = reference.attention_step_inputs()
    device = step.hidden_ctx.device.type
    layer_f32 = reference.build_hf_attention(seed=reference.WEIGHT_SEED, device=device).float()
    oracle_f32 = reference.decode_reference(
        layer_f32, step.hidden_ctx.float(), step.hidden_new.float()
    )
    oracle_gap = (reference.attention_step_oracle(step).float() - oracle_f32).abs().max().item()
    _grown, entry = _entry(step)

    contract.compared(
        tf,
        tmp_path,
        shipped_source(MODEL),
        CASES[0],
        ATTENTION,
        activations=step.args,
        weights=step.weights,
        expected=(oracle_f32.to(reference.DTYPE), entry),
        held=(
            ("allclose", {"atol": oracle_gap, "rtol": 0.0}),
            ("allclose", {"atol": _fp8_step(entry), "rtol": 0.0}),
        ),
    )


@cuda_only
@pytest.mark.parametrize("ctx_len", CTX_LENGTHS)
def test_the_step_is_authored_over_a_range_of_context_lengths(
    tf, shipped_source, tmp_path, ctx_len
) -> None:
    """The same description, at two context lengths, each against its own oracle.

    `ctx_len` is a range rather than the one number the step was written at, and a
    step that had baked a length in would agree at one length only.
    """
    step = reference.attention_step_inputs(ctx_len=ctx_len)
    want = reference.attention_step_oracle(step)

    assert step.kv_cache.shape[1] == ctx_len
    _asked(
        tf,
        tmp_path,
        shipped_source(MODEL),
        step,
        out_held=("allclose", {"atol": _bf16_ulps(want), "rtol": 0.0}),
    )


@cuda_only
def test_the_step_returns_the_cache_entry_to_append(tf, shipped_source, tmp_path) -> None:
    """The returned latent is this token's cache entry.

    The returned latent is this token's cache entry: appending it to the cache the
    step was given reproduces the cache a context one token longer holds.

    Checked against a rebuilt cache rather than against the step's own input, so a
    step that returned its input unchanged would fail. The two agree to the fp8 grid
    the latent is stored on and no closer, which is the point of storing it that way.
    """
    step = reference.attention_step_inputs()
    grown, entry = _entry(step)
    want = reference.attention_step_oracle(step)

    assert grown.shape[1] == step.ctx_len + 1
    assert torch.equal(grown[:, : step.ctx_len], step.kv_cache)
    _asked(
        tf,
        tmp_path,
        shipped_source(MODEL),
        step,
        out_held=("allclose", {"atol": _bf16_ulps(want), "rtol": 0.0}),
    )
