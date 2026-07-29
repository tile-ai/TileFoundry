"""Qwen3.5-35B-A3B's Gated DeltaNet decode step against Hugging Face's own.

Nothing here is specialised at a length, because nothing here carries one: this
mixer's state is fixed-size, so every extent in the kernel is a literal. It is
still evaluated at two context lengths, because the *state* it is handed depends
on the context even though its shape does not -- a step that quietly ignored the
recurrent matrix would agree at one length and not at two.

The mixer's weights are declared ``ConstTensor``, so they are bound once by the
loading and every call below passes the hidden state and the state alone. The
convolution window and the recurrent matrix are among what is passed: they are
what the step is handed, not what it holds.
"""
from __future__ import annotations

import torch

from tests.models.qwen3_5_35b_a3b import reference
from tests.models.qwen3_5_35b_a3b.model import advance_state

DEV = reference.DEVICE
ATOL = RTOL = 2e-5

#: Measured on an H200, f32. Mixer output: 4.06e-07 at ctx_len 25 (reference
#: maximum magnitude 0.436) and 5.66e-07 at 41 (0.466). Recurrent state:
#: 1.34e-07 / 4.32e-07 (state magnitude 0.186 / 0.379). Convolution column:
#: 3.81e-06 / 5.25e-06, on columns of magnitude about 10 -- looser than the
#: others in absolute terms and tighter in relative terms, which is what a value
#: that has been through one projection and no reduction should look like.
MEASURED_MIXER_MAX_ABS_DIFF = 1e-06
MEASURED_STATE_MAX_ABS_DIFF = 1e-06
MEASURED_CONV_MAX_ABS_DIFF = 8e-06

CTX_LENGTHS = (25, 41)


def test_linear_attention_matches_hugging_face():
    """linear_attention (input_layernorm + `Qwen3_5MoeGatedDeltaNet`: causal
    convolution, L2-normalised query and key, the gated delta rule and the gated
    output norm) vs HF's own mixer at the decoded position, at two lengths."""
    for ctx_len in CTX_LENGTHS:
        step = reference.linear_step(ctx_len=ctx_len, device=DEV)
        loaded = reference.load_mixer("linear_attention", step.layer)
        out, _entry, _state = loaded.linear_attention(step.hidden_new, *step.mixer_acts)

        want = reference.linear_mixer_oracle(step)
        difference = (out.float() - want.float()).abs().max().item()
        assert difference <= MEASURED_MIXER_MAX_ABS_DIFF, (ctx_len, difference)
        torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_the_step_returns_the_state_to_carry_forward():
    """The step's returned convolution column and recurrent matrix are the state
    a caller should hold afterwards: sliding the window and taking the matrix
    reproduces the state a context one token longer would have produced.

    Checked against a rebuilt state rather than against the step's own inputs, so
    a step that returned its inputs unchanged would fail -- which for the
    recurrent matrix is the failure worth guarding, since its shape gives nothing
    away. The slide itself is the decoder's own ``advance_state``, so the rule is
    stated once.
    """
    for ctx_len in CTX_LENGTHS:
        step = reference.linear_step(ctx_len=ctx_len, device=DEV)
        loaded = reference.load_mixer("linear_attention", step.layer)
        _out, entry, state = loaded.linear_attention(step.hidden_new, *step.mixer_acts)

        want_conv, want_state = reference.advanced_state_oracle(step)
        slid, state = advance_state(
            "linear_attention", (step.conv_state, step.recurrent_state), (entry, state)
        )

        assert tuple(slid.shape) == tuple(want_conv.shape)
        assert tuple(state.shape) == tuple(want_state.shape)
        conv_difference = (slid.float() - want_conv.float()).abs().max().item()
        state_difference = (state.float() - want_state.float()).abs().max().item()
        assert conv_difference <= MEASURED_CONV_MAX_ABS_DIFF, (ctx_len, conv_difference)
        assert state_difference <= MEASURED_STATE_MAX_ABS_DIFF, (ctx_len, state_difference)
        torch.testing.assert_close(slid.float(), want_conv.float(), atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(state.float(), want_state.float(), atol=ATOL, rtol=RTOL)


def test_the_prior_state_is_read():
    """The recurrent matrix handed in reaches the answer.

    A linear-attention step has no ``ctx_len`` in its signature, so nothing about
    its shape says it consulted the context at all -- an implementation that
    dropped the incoming matrix would produce a plausible tensor of the right
    size. Measured by zeroing the matrix: if that does not move the output, the
    kernel is not reading it, and every agreement above would be an agreement
    about one token in isolation.
    """
    step = reference.linear_step(device=DEV)
    loaded = reference.load_mixer("linear_attention", step.layer)
    out, _entry, _state = loaded.linear_attention(step.hidden_new, *step.mixer_acts)

    stateless, _entry, _state = loaded.linear_attention(
        step.hidden_new, step.conv_state, torch.zeros_like(step.recurrent_state)
    )

    moved = (out.float() - stateless.float()).abs().max().item()
    assert moved > 100 * ATOL, (
        f"zeroing the recurrent state moved the answer by only {moved}, so this "
        f"boundary does not distinguish a kernel that ignores its history"
    )


def test_the_convolution_window_is_read():
    """The convolution's left context reaches the answer.

    The same argument as the recurrent matrix, for the other half of the state:
    at one token per step, a kernel that convolved only the current column would
    be a kernel with a kernel size of one, and nothing about its output shape
    would say so.
    """
    step = reference.linear_step(device=DEV)
    loaded = reference.load_mixer("linear_attention", step.layer)
    out, _entry, _state = loaded.linear_attention(step.hidden_new, *step.mixer_acts)

    windowless, _entry, _state = loaded.linear_attention(
        step.hidden_new, torch.zeros_like(step.conv_state), step.recurrent_state
    )

    moved = (out.float() - windowless.float()).abs().max().item()
    assert moved > 100 * ATOL, (
        f"zeroing the convolution window moved the answer by only {moved}, so "
        f"this boundary does not distinguish a kernel with no left context"
    )


def test_the_state_decays_rather_than_accumulating():
    """``g`` is negative, so ``exp(g)`` is a decay strictly inside (0, 1).

    That is what stops the recurrent matrix from growing without a token asking
    it to, and it is a property of the *sign* of ``-exp(A_log) * softplus(...)``
    rather than of any weight.

    Also measured: that the drawn step actually spans that interval rather than
    sitting at one end of it. A fixture whose every head retained nearly
    everything would satisfy the sign condition and exercise none of the gating
    the layer is named for; the drawn weights put some heads below 0.01 and some
    above 0.99, so both extremes are on the path the comparisons above take.
    """
    step = reference.linear_step(device=DEV)
    mixer = step.layer.linear_attn
    with torch.no_grad():
        normed = step.layer.input_layernorm(step.hidden_new)
        decay = (
            -mixer.A_log.float().exp()
            * torch.nn.functional.softplus(
                mixer.in_proj_a(normed).float() + mixer.dt_bias
            )
        ).exp()

    assert decay.max().item() < 1.0
    assert decay.min().item() > 0.0
    assert decay.min().item() < 0.1, (
        f"the drawn step's weakest retention is {decay.min().item()}; no head "
        f"forgets, so the decay is not being exercised"
    )
    assert decay.max().item() > 0.9, (
        f"the drawn step's strongest retention is {decay.max().item()}; no head "
        f"remembers, so the recurrence is not being exercised"
    )
