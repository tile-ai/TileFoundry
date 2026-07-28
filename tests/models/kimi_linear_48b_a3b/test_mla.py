"""Kimi-Linear MLA decode step vs Hugging Face, in both of its forms.

cpu + f32, because the oracle is a CPU f32 baseline and this submodule is small
enough not to need a device.

Every number this file asserts was measured before it was written down. The
parity tests are only half the point: the perturbation tests below establish that
the parity tests can *fail*, which is what makes them evidence. Without them a
kernel that ignored the cache, or used the wrong scaling, would pass a shape
check and read as correct.
"""
from __future__ import annotations

import torch

from tests.models.kimi_linear_48b_a3b import config, reference
from tests.models.kimi_linear_48b_a3b import submodules as kimi
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.specialize import specialize_concretely

DEV = "cpu"
ATOL = RTOL = 2e-4

#: Two lengths, so a kernel that only works at the length it was authored against
#: cannot pass. Neither is a multiple of the 32 heads.
CTX_LENGTHS = (24, 40)

#: What a perturbed run has to move the output by before the corresponding parity
#: test counts as discriminating. Four orders of magnitude above the f32
#: round-off the parity tests accept, so "it changed" cannot be round-off.
DISCRIMINATION = 1e-3


def _run(drawn, args=None):
    """`mla_attention` specialised at *drawn*'s context length and evaluated."""
    fn = specialize_concretely(kimi.mla_attention, {"ctx_len": drawn.ctx_len})
    return evaluate(fn, *(args if args is not None else drawn.args), device=DEV)


def test_identity_rotary_is_exactly_the_identity():
    """`cos = 1, sin = 0` leaves q and k untouched -- exactly, not approximately.

    This is the whole basis for expressing `mla_use_nope: true` with the same
    attention module rather than a second one, so it is measured rather than
    argued: `apply_rotary_pos_emb(x, x, 1, 0)` is `x * 1 + rotate_half(x) * 0`.
    Measured max abs diff 0.0 on both.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    rope_dim = config.REAL.qk_rope_head_dim
    torch.manual_seed(0)
    q = torch.randn(1, 4, 7, rope_dim)
    k = torch.randn(1, 1, 7, rope_dim)
    q_out, k_out = apply_rotary_pos_emb(
        q, k, torch.ones(1, 7, rope_dim), torch.zeros(1, 7, rope_dim)
    )

    assert (q_out - q).abs().max().item() == 0.0
    assert (k_out - k).abs().max().item() == 0.0


def test_mla_nope_matches_hf():
    """Kimi's own MLA form: NoPE, at two context lengths.

    NoPE does not drop the 64 rotary dimensions -- it stops rotating them. They
    still enter the score and the `qk_head_dim = 192` scaling denominator, which
    is why `test_mla_scaling_is_qk_head_dim` below can tell the difference.
    """
    for ctx_len in CTX_LENGTHS:
        drawn = reference.mla_step_inputs(ctx_len=ctx_len, device=DEV, nope=True)
        out, _k, _v = _run(drawn)
        want = reference.mla_step_oracle(drawn)
        torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_mla_rope_matches_hf():
    """The same kernel with a real rotary, at two context lengths.

    Not a configuration Kimi ships -- it is NoPE -- but it exercises the rotary
    path of the same kernel, and its agreement is what shows `tf.rope` and the
    oracle share the rotate-half convention (`rope_interleave=False`).
    """
    for ctx_len in CTX_LENGTHS:
        drawn = reference.mla_step_inputs(ctx_len=ctx_len, device=DEV, nope=False)
        out, _k, _v = _run(drawn)
        want = reference.mla_step_oracle(drawn)
        torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_mla_returns_the_cache_entry_to_append():
    """The returned key and value are this token's cache entry.

    Checked against a cache rebuilt over the context with the token appended, not
    against the step's own inputs, so a step that echoed its inputs would fail.
    """
    drawn = reference.mla_step_inputs(device=DEV, nope=True)
    _out, k_new, v_new = _run(drawn)

    want_k, want_v = reference.mla_appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert tuple(grown_k.shape) == tuple(want_k.shape)
    assert tuple(grown_v.shape) == tuple(want_v.shape)
    torch.testing.assert_close(grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL)


def test_mla_scaling_is_qk_head_dim_not_v_head_dim():
    """`qk_head_dim ** -0.5`, and the plausible wrong guess is detectable.

    Nothing in the published config says which dimension the score is scaled by,
    and `v_head_dim ** -0.5` is the natural guess: 0.0883883 against the correct
    0.0721688, 22.5% apart. Substituting it moves the output by 4.4e-02, three
    orders of magnitude above the 6.9e-07 the correct value achieves -- so this
    test is what stops the parity tests above from passing on a wrong constant.
    """
    drawn = reference.mla_step_inputs(device=DEV, nope=True)
    want = reference.mla_step_oracle(drawn)

    args = list(drawn.args)
    args[11] = torch.full((1, 1, 1, 1), config.REAL.v_head_dim ** -0.5, device=DEV)
    wrong, _k, _v = _run(drawn, args)

    assert (wrong.float() - want.float()).abs().max().item() > DISCRIMINATION


def test_mla_cache_pairing_is_load_bearing():
    """Permuting one side of the cache breaks the answer.

    Softmax attention over a cache is permutation-invariant if both sides are
    permuted together -- position is already baked into the stored key -- so a
    joint permutation would prove nothing. Permuting the keys *without* the values
    breaks the pairing between them, and that must show: measured 6.5e-01 for keys
    and 5.7e-01 for values. A kernel that read the cache at fixed offsets, or
    ignored it, would fail here.
    """
    drawn = reference.mla_step_inputs(device=DEV, nope=True)
    want = reference.mla_step_oracle(drawn)
    torch.manual_seed(0)
    perm = torch.randperm(drawn.ctx_len)

    args = list(drawn.args)
    args[9] = drawn.k_cache[:, perm]
    keys_shuffled, _k, _v = _run(drawn, args)
    assert (keys_shuffled.float() - want.float()).abs().max().item() > DISCRIMINATION

    args = list(drawn.args)
    args[10] = drawn.v_cache[:, perm]
    values_shuffled, _k, _v = _run(drawn, args)
    assert (values_shuffled.float() - want.float()).abs().max().item() > DISCRIMINATION


def test_nope_and_rope_are_different_functions():
    """The two forms disagree, so neither test above is passing vacuously.

    If the rotary were being ignored -- caches unread, `pos_ids` dropped -- both
    parity tests would pass against the same oracle and neither would notice.
    Measured separation 4.0e-01.
    """
    nope = reference.mla_step_oracle(reference.mla_step_inputs(device=DEV, nope=True))
    rope = reference.mla_step_oracle(reference.mla_step_inputs(device=DEV, nope=False))
    assert (nope.float() - rope.float()).abs().max().item() > DISCRIMINATION
