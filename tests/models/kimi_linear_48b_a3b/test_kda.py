"""Kimi Delta Attention: what can be established without an oracle, and the
block recording that its values cannot.

Read the assertions here for what they are. **None of them checks a value.** The
KDA reference is blocked (`reference.KDA_BLOCK_REASON`), so nothing in this file
claims `kda_attention` computes the right thing -- only that it is well-formed,
that it executes, that its state is shaped and typed the way the decode contract
requires, and that the two properties which are true by construction rather than
by transcription really hold.

That distinction is the point. A file of shape assertions labelled as if it were
a correctness suite would be exactly the substitution of a convenient proxy for
the real property that this package is trying not to make.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.kimi_linear_48b_a3b import case, config, reference
from tests.models.kimi_linear_48b_a3b import submodules as kimi
from tilefoundry.evaluator import evaluate

DEV = "cpu"
SHAPE = config.REAL


def _kda_args(seed: int = 0, device: str = DEV):
    """The reference's own argument draw, as a list so tests can perturb one entry.

    One definition, shared with `reference.kda_step_inputs`, so a signature change
    cannot leave this file agreeing with a stale order.
    """
    return list(reference.kda_step_inputs(device=device, seed=seed).args)


def test_kda_reference_is_blocked_and_the_gate_holds_it():
    """The boundary runs, cannot be scored, and the gate records exactly that.

    Three separate claims, because the block is only worth anything if all three
    hold: the run really happens (`run_kda_step` evaluates the whole layer before
    it gives up), the failure carries the recorded reason, and `CapabilityGate`
    accepts it as the stated limit. A gate that would accept *any* failure would
    let a second, unrelated defect hide behind this one.
    """
    drawn = reference.kda_step_inputs()

    with pytest.raises(AssertionError) as raised:
        reference.run_kda_step(drawn)
    assert reference.KDA_BLOCK_REASON in str(raised.value)

    gate = case.CASE.reference.gate
    assert gate.blocked
    with pytest.raises(AssertionError):
        gate.hold(
            lambda: reference.run_kda_step(drawn),
            expect=AssertionError,
            label="kda",
        )


def test_a_block_for_another_reason_would_not_be_accepted():
    """The gate rejects a failure that is not the one it records.

    This is what stops the block from becoming a catch-all: if KDA later broke for
    an unrelated reason, the gate must complain rather than report the recorded
    limit.
    """
    from tests.models.corpus import CorpusError  # noqa: PLC0415

    gate = case.CASE.reference.gate

    def fails_differently():
        raise AssertionError("something else entirely")

    with pytest.raises(CorpusError):
        gate.hold(fails_differently, expect=AssertionError, label="kda")


def test_kda_state_carries_no_ctx_len():
    """Every KDA parameter and result is a fixed shape.

    This is the contract claim the module docstring makes, checked rather than
    asserted in prose: a KDA layer's recurrent state is not a growing per-position
    cache, so `ctx_len` must not appear anywhere in its signature. A model with a
    dynamic dimension hiding in the state would be a different claim about the
    decode contract than the one recorded.
    """
    signature = kimi.kda_attention.type
    for parameter in signature.parameters:
        assert "ctx_len" not in str(parameter)
    assert "ctx_len" not in str(signature.return_type)


def test_kda_state_and_window_shapes_are_the_published_ones():
    """The state is `[heads, v_dim, k_dim]` and the window is `kernel - 1` deep.

    32 x 128 x 128 for the delta-rule memory and 3 x 4096 for each of the three
    convolution windows, which is what vLLM's `kda_state_shape` computes from the
    same published fields. Top-level `head_dim: 72` appears in none of them: KDA
    reads `linear_attn_config.head_dim`, which is 128.
    """
    out, state, conv_q, conv_k, conv_v = evaluate(
        kimi.kda_attention, *_kda_args(), device=DEV
    )

    assert tuple(out.shape) == (1, config.SEQ_LEN, SHAPE.hidden)
    assert tuple(state.shape) == (1, SHAPE.kda_heads, SHAPE.kda_head_dim, SHAPE.kda_head_dim)
    for window in (conv_q, conv_k, conv_v):
        assert tuple(window.shape) == (1, SHAPE.short_conv - 1, SHAPE.kda_proj)
    assert SHAPE.kda_head_dim == 128 != SHAPE.hidden // SHAPE.n_heads


def test_kda_executes_and_stays_finite():
    """The kernel runs end to end and produces finite numbers.

    Not a correctness claim. It rules out the failure modes that would make the
    blocked reference impossible to lift later -- a malformed graph, a shape that
    only type-checks symbolically, an exp or softplus that overflows on ordinary
    inputs -- without pretending to check what the numbers are.
    """
    out, state, *windows = evaluate(kimi.kda_attention, *_kda_args(), device=DEV)

    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()
    for window in windows:
        assert torch.isfinite(window).all()


def test_short_conv_window_shifts_by_exactly_one_position():
    """The convolution window returned is the input window with this token
    appended and the oldest position dropped.

    True by construction rather than by transcription from any implementation, so
    it is checkable with no oracle: the stored window is a slice of
    `concat(state, x)`, and appending then evicting is what a depthwise causal
    convolution of kernel 4 needs its caller to hold. Measured exact (0.0).
    """
    torch.manual_seed(1)
    window = SHAPE.short_conv - 1
    x = torch.randn(1, config.SEQ_LEN, SHAPE.kda_proj) * 0.05
    conv_w = torch.randn(SHAPE.short_conv, SHAPE.kda_proj) * 0.05
    state = torch.randn(1, window, SHAPE.kda_proj) * 0.05

    _out, next_state = evaluate(kimi.short_conv, x, conv_w, state, device=DEV)

    want = torch.cat([state, x], dim=1)[:, 1:]
    assert (next_state - want).abs().max().item() == 0.0


def test_kda_state_is_load_bearing():
    """A different incoming state gives a different answer.

    Weak on purpose -- with no oracle this cannot say the state is used
    *correctly* -- but it does rule out a kernel that accepted the state and
    ignored it, which is the failure a shape check cannot see and which would make
    every other assertion in this file pass.
    """
    args = _kda_args()
    base_out, base_state, *_ = evaluate(kimi.kda_attention, *args, device=DEV)

    perturbed = list(args)
    perturbed[20] = args[20] + 1.0
    other_out, other_state, *_ = evaluate(kimi.kda_attention, *perturbed, device=DEV)

    assert (other_out - base_out).abs().max().item() > 1e-3
    assert (other_state - base_state).abs().max().item() > 1e-3


def test_layer_taxonomy_is_one_based():
    """The published layer lists are 1-based, and the minimum model reflects it.

    Recorded as a test because the config contradicts the natural reading: layer 0
    is KDA and layer 3 is the first MLA layer, and with `first_k_dense_replace: 1`
    layer 0 is the only one with a dense MLP. A two-layer minimum of layers 0 and 1
    would therefore contain two KDA layers and no MLA one.
    """
    assert config.is_kda_layer(0)
    assert config.is_kda_layer(1)
    assert config.is_kda_layer(2)
    assert not config.is_kda_layer(3)

    assert config.is_dense_layer(0)
    assert not config.is_dense_layer(1)

    kda, mla = config.MINIMUM_LAYERS
    assert config.is_kda_layer(kda) and config.is_dense_layer(kda)
    assert not config.is_kda_layer(mla) and not config.is_dense_layer(mla)


def test_depended_config_fields_match_their_recorded_digest():
    """The fields this package rests on are the ones it was written against.

    A digest over just those fields, not the whole file: an upstream edit to a
    field nobody here reads should not invalidate the model, and an edit to one of
    these must.
    """
    assert config.depended_digest() == config.DEPENDED_SHA256
    assert config.DEPENDED_FIELDS["model_type"] == "kimi_linear"
    assert config.SOURCE_REVISION == "e1df551a447157d4658b573f9a695d57658590e9"
