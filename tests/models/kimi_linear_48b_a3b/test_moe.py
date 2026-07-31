"""Kimi-Linear MoE vs Hugging Face's `DeepseekV3MoE` at Kimi's numbers.

CUDA, because the expert weights are large: 256 experts is several gigabytes, and
the evaluated run peaks near 17 GB.

Why DeepSeek-V3 is a legitimate oracle here, rather than a lookalike: at
`n_group = topk_group = 1` its grouped top-k is the identity, and its sigmoid
router applies `routed_scaling_factor` to the *routing weights* while vLLM's
`KimiMoE` applies it to the *expert output*. Those are the same function -- the
expert combine is linear in the weights -- measured at 7.1e-08 on an output of
magnitude 1.156, which is the rounding of the weight product and not a
difference in what is computed.

Most tests run at a reduced expert count: `SMALL_MOE` is the published config with
a quarter of the experts, built from the same source, so it is the same kernel at
a size that fits beside seven other workers. The headline parity test runs at the
published 256.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.decode_oracle import agrees_as_a_component
from tests.models.kimi_linear_48b_a3b import reference
from tests.models.kimi_linear_48b_a3b.model import (
    KimiLinear48BA3B,
    build_kimi_linear_48b_a3b,
)
from tests.models.kimi_linear_48b_a3b.reference import SMALL_MOE
from tilefoundry.evaluator import evaluate

CONFIG = reference.CONFIG

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


#: Expert count for the tests that only need to tell right from wrong. Eight times
#: `top_k`, so a top-8 still selects a small minority of the experts.
SMALL_EXPERTS = SMALL_MOE.num_experts

#: Four orders of magnitude above the round-off the parity tests accept.
DISCRIMINATION = 1e-3

#: Headroom each expert count needs, measured: 256 experts peak at 17.1 GB.
_NEEDED_GB = {CONFIG.num_experts: 24.0, SMALL_EXPERTS: 8.0}


def _moe_at(config_at) -> object:
    """This model's MoE kernel at *config_at*, from the one public builder."""
    return build_kimi_linear_48b_a3b(config_at).moe.lookup("moe")


def _router_at(config_at) -> object:
    """Its router, the same way."""
    return build_kimi_linear_48b_a3b(config_at).moe.lookup("router")


def _device(n_experts: int) -> str:
    """The CUDA device with room for *n_experts*, or skip.

    This box has several GPUs and is shared, so a fixed `"cuda"` makes the test's
    result depend on what else happens to be resident on device 0. Choosing by
    free memory -- and skipping rather than failing when none has room -- keeps an
    OOM from being reported as a disagreement with the oracle, which is a
    different fact entirely.
    """
    needed = _NEEDED_GB[n_experts] * 1e9
    best, best_free = None, 0
    for index in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(index)
        if free > best_free:
            best, best_free = index, free
    if best is None or best_free < needed:
        pytest.skip(
            f"no CUDA device with {needed / 1e9:.0f} GB free for {n_experts} "
            f"experts (most free: {best_free / 1e9:.1f} GB)"
        )
    return f"cuda:{best}"


@pytest.fixture(scope="module")
def small_moe():
    """One reduced-count Hugging Face MoE and the device it lives on.

    Module-scoped because building it is most of the cost, and the tests only ever
    read it.
    """
    device = _device(SMALL_EXPERTS)
    return reference.build_hf_moe(device=device, n_experts=SMALL_EXPERTS), device


def _small(act_seed, small_moe):
    hf_moe, device = small_moe
    return reference.moe_inputs(
        device=device, act_seed=act_seed, hf_moe=hf_moe, n_experts=SMALL_EXPERTS
    ), device


def test_moe_matches_hf_at_published_expert_count():
    """The full 256-expert MoE, over four independent draws.

    Four draws rather than one batch of four tokens: the decode contract fixes the
    token count at the literal 1, so breadth over which experts get selected has
    to come from redrawing. The four draws select genuinely different expert sets
    (measured: 8-expert selections overlapping in 4 to 6 members, not identical).
    """
    device = _device(CONFIG.num_experts)
    hf_moe = reference.build_hf_moe(device=device)
    try:
        for act_seed in reference.MOE_DRAWS:
            drawn = reference.moe_inputs(
                device=device, act_seed=act_seed, hf_moe=hf_moe
            )
            out = evaluate(KimiLinear48BA3B.moe.lookup("moe"), *drawn.args, device=device)
            want = reference.moe_oracle(drawn)
            agrees_as_a_component(out, want)
    finally:
        del hf_moe
        torch.cuda.empty_cache()


def test_moe_matches_hf_at_reduced_expert_count(small_moe):
    """The same source at `SMALL_EXPERTS`, so the reduced count the perturbation
    tests below use is itself known good rather than assumed."""
    drawn, device = _small(reference.ACTIVATION_SEED, small_moe)
    out = evaluate(_moe_at(SMALL_MOE), *drawn.args, device=device)
    want = reference.moe_oracle(drawn)
    agrees_as_a_component(out, want)


def test_router_bias_is_load_bearing(small_moe):
    """Zeroing `e_score_correction_bias` changes the answer.

    This is the reason `build_hf_moe` draws the bias nonzero, and the reason that
    must not be "simplified" away. The router selects on `sigmoid(logits) + bias`
    but takes the routing weights from the *unbiased* scores. At bias = 0 those are
    the same tensor, so an implementation that gathered the biased scores would be
    indistinguishable from a correct one -- and the whole select-with-bias /
    gather-without-bias distinction would go untested.

    Measured: zeroing the bias moves the output by 1.55e+00.
    """
    drawn, device = _small(reference.ACTIVATION_SEED, small_moe)
    want = reference.moe_oracle(drawn)

    args = list(drawn.args)
    args[3] = torch.zeros_like(drawn.args[3])
    unbiased = evaluate(_moe_at(SMALL_MOE), *args, device=device)

    assert (unbiased.float() - want.float()).abs().max().item() > DISCRIMINATION


def test_router_gathers_unbiased_scores(small_moe):
    """The routing weights come from the unbiased sigmoid scores.

    A direct check on the one routing subtlety no config agrees or disagrees about,
    against a hand-computed *wrong* variant rather than against an oracle: if the
    weights were gathered from `scores + bias` instead, they would differ by
    1.08e-01 measured. So this asserts two things at once -- that the HIR matches
    the correct variant, and that the correct and wrong variants are far enough
    apart for the first assertion to mean something.
    """
    drawn, device = _small(reference.ACTIVATION_SEED, small_moe)
    tokens = drawn.normed.view(-1, CONFIG.hidden_size)
    bias = drawn.args[3]

    with torch.no_grad():
        scores = (tokens.float() @ drawn.args[2].float()).sigmoid()
        biased = scores + bias.float()
        _v, indices = biased.topk(CONFIG.num_experts_per_token, dim=-1)

        def normalise(gathered):
            weights = gathered / (gathered.sum(-1, keepdim=True) + 1e-20)
            return (weights * CONFIG.routed_scaling_factor).to(reference.DTYPE)

        right = normalise(scores.gather(1, indices))
        wrong = normalise(biased.gather(1, indices))

    weights, hir_indices = evaluate(
        _router_at(SMALL_MOE), tokens, drawn.args[2], bias, drawn.args[4], device=device
    )

    # Same experts, and the weights are the unbiased ones.
    assert sorted(hir_indices[0].tolist()) == sorted(indices[0].tolist())
    agrees_as_a_component(weights.sort(-1).values, right.sort(-1).values)
    assert (right - wrong).abs().max().item() > DISCRIMINATION


def test_routed_scaling_is_applied_after_normalisation(small_moe):
    """`moe_renormalize: true` means normalise, *then* scale.

    The order is observable because the two orders are not the same function:
    scaling the selected scores before dividing by their sum cancels the factor
    entirely, leaving the same output as a scaling factor of 1.0. Measured: a
    scaling factor of 1.0 moves the output by 7.4e-01 from the correct 2.446, so
    an implementation that folded the factor into the denominator would be caught
    here.
    """
    drawn, device = _small(reference.ACTIVATION_SEED, small_moe)
    want = reference.moe_oracle(drawn)

    args = list(drawn.args)
    args[4] = torch.full((1, 1), 1.0, device=device, dtype=reference.DTYPE)
    unscaled = evaluate(_moe_at(SMALL_MOE), *args, device=device)

    assert (unscaled.float() - want.float()).abs().max().item() > DISCRIMINATION


def test_shared_expert_contributes(small_moe):
    """The shared expert is really in the sum.

    `num_shared_experts: 1`, and it is unscaled -- `routed_scaling_factor` applies
    to the routed branch only. Zeroing its gate projection moves the output by
    1.09e+00, so a kernel that dropped the shared branch could not pass the parity
    test above.
    """
    drawn, device = _small(reference.ACTIVATION_SEED, small_moe)
    want = reference.moe_oracle(drawn)

    args = list(drawn.args)
    args[8] = torch.zeros_like(drawn.args[8])
    without = evaluate(_moe_at(SMALL_MOE), *args, device=device)

    assert (without.float() - want.float()).abs().max().item() > DISCRIMINATION
