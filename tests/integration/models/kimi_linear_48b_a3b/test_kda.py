"""State what Kimi Delta Attention establishes without an oracle.

The blocked reference means these tests claim no value correctness. They cover
well-formed execution, decode state shapes and dtypes, and properties true by
construction. Keeping that boundary explicit prevents configuration assertions
from masquerading as a correctness suite.
"""

from __future__ import annotations

import pytest
import torch

from tests.models.decode_oracle import SEQ_LEN
from tests.models.kimi_linear_48b_a3b import case, reference
from tests.models.kimi_linear_48b_a3b.model import KimiLinear48BA3B
from tests.models.kimi_linear_48b_a3b.reference import (
    KDA_HEAD_DIM,
    KDA_NUM_HEADS,
    KDA_PROJ,
    SHORT_CONV_KERNEL_SIZE,
)
from tilefoundry.evaluator import evaluate

DEV = "cpu"
CONFIG = reference.CONFIG


MINIMUM_LAYERS = (0, 3)


def _is_dense_layer(layer_idx: int) -> bool:
    """Whether 0-based *layer_idx* uses a dense MLP rather than the MoE."""
    return layer_idx < CONFIG.first_k_dense_replace


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
    """Every KDA parameter and result is a fixed config.

    This is the contract claim the module docstring makes, checked rather than
    asserted in prose: a KDA layer's recurrent state is not a growing per-position
    cache, so `ctx_len` must not appear anywhere in its signature. A model with a
    dynamic dimension hiding in the state would be a different claim about the
    decode contract than the one recorded.
    """
    signature = KimiLinear48BA3B.kda.lookup("kda_attention").type
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
        KimiLinear48BA3B.kda.lookup("kda_attention"), *_kda_args(), device=DEV
    )

    assert tuple(out.shape) == (1, SEQ_LEN, CONFIG.hidden_size)
    assert tuple(state.shape) == (1, KDA_NUM_HEADS, KDA_HEAD_DIM, KDA_HEAD_DIM)
    for window in (conv_q, conv_k, conv_v):
        assert tuple(window.shape) == (1, SHORT_CONV_KERNEL_SIZE - 1, KDA_PROJ)
    assert KDA_HEAD_DIM == 128 != CONFIG.hidden_size // CONFIG.num_attention_heads


def test_kda_executes_and_stays_finite():
    """The kernel runs end to end and produces finite numbers.

    Not a correctness claim. It rules out the failure modes that would make the
    blocked reference impossible to lift later -- a malformed graph, a config that
    only type-checks symbolically, an exp or softplus that overflows on ordinary
    inputs -- without pretending to check what the numbers are.
    """
    out, state, *windows = evaluate(
        KimiLinear48BA3B.kda.lookup("kda_attention"), *_kda_args(), device=DEV
    )

    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()
    for window in windows:
        assert torch.isfinite(window).all()


def test_short_conv_window_shifts_by_exactly_one_position():
    """Test short conv window shifts by exactly one position.

    The returned window appends this token and drops the oldest position. This is
    checkable without an oracle because it is a slice of ``concat(state, x)``.
    The claim concerns positions, not arithmetic, so it remains exact at bf16 and
    uses equality rather than a tolerance.
    """
    torch.manual_seed(1)
    window = SHORT_CONV_KERNEL_SIZE - 1
    x = (torch.randn(1, SEQ_LEN, KDA_PROJ) * 0.05).to(reference.DTYPE)
    conv_w = (torch.randn(SHORT_CONV_KERNEL_SIZE, KDA_PROJ) * 0.05).to(reference.DTYPE)
    state = (torch.randn(1, window, KDA_PROJ) * 0.05).to(reference.DTYPE)

    _out, next_state = evaluate(
        KimiLinear48BA3B.kda.lookup("short_conv"), x, conv_w, state, device=DEV
    )

    want = torch.cat([state, x], dim=1)[:, 1:].to(next_state.dtype)
    assert (next_state - want).abs().max().item() == 0.0


def test_kda_state_is_load_bearing():
    """A different incoming state gives a different answer.

    Weak on purpose -- with no oracle this cannot say the state is used
    *correctly* -- but it does rule out a kernel that accepted the state and
    ignored it, which is the failure a config check cannot see and which would make
    every other assertion in this file pass.
    """
    args = _kda_args()
    base_out, base_state, *_ = evaluate(
        KimiLinear48BA3B.kda.lookup("kda_attention"), *args, device=DEV
    )

    perturbed = list(args)
    perturbed[20] = args[20] + 1.0
    other_out, other_state, *_ = evaluate(
        KimiLinear48BA3B.kda.lookup("kda_attention"), *perturbed, device=DEV
    )

    assert (other_out - base_out).abs().max().item() > 1e-3
    assert (other_state - base_state).abs().max().item() > 1e-3


def test_layer_taxonomy_is_one_based():
    """The published layer lists are 1-based, and the minimum model reflects it.

    Recorded as a test because the config contradicts the natural reading: layer 0
    is KDA and layer 3 is the first MLA layer, and with `first_k_dense_replace: 1`
    layer 0 is the only one with a dense MLP. A two-layer minimum of layers 0 and 1
    would therefore contain two KDA layers and no MLA one.

    Asked of `KimiLinearConfig.is_kda_layer` -- the checkpoint's own method, from
    the config class vendored into `model.py` -- rather than of a reimplementation
    of it, so what is checked is the published reading of the published lists.
    """
    assert CONFIG.model_type == "kimi_linear"

    assert CONFIG.is_kda_layer(0)
    assert CONFIG.is_kda_layer(1)
    assert CONFIG.is_kda_layer(2)
    assert not CONFIG.is_kda_layer(3)

    assert _is_dense_layer(0)
    assert not _is_dense_layer(1)

    kda, mla = MINIMUM_LAYERS
    assert CONFIG.is_kda_layer(kda) and _is_dense_layer(kda)
    assert not CONFIG.is_kda_layer(mla) and not _is_dense_layer(mla)
