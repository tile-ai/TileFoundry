"""Qwen3.5-35B-A3B's full-attention decode step against Hugging Face's own.

The kernel carries ``ctx_len`` as a range, and it is run as it is authored: the
loading evaluates the published Function and the range is bound from the cache it
is handed, so what is measured is the kernel the model declares rather than a copy
of it resolved at one length. The corpus runs the same loading.

Weights are declared ``ConstTensor``, so they are bound by the loading and every
call below passes activations alone; those come from ``reference.py``'s drawn step
rather than being assembled here, so a signature change cannot leave one test
agreeing with a stale order.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from tests.models.decode_oracle import agrees_as_a_component
from tests.models.qwen3_5_35b_a3b import reference
from tests.models.qwen3_5_35b_a3b.model import advance_state

DEV = reference.DEVICE

#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither divides either head count, and neither is a
#: multiple of the other.
CTX_LENGTHS = (25, 41)



def test_full_attention_matches_hugging_face():
    """full_attention (input_layernorm + `Qwen3_5MoeAttention`: GQA, per-head
    q_norm/k_norm, partial RoPE and the output gate, over the cache and the new
    token) vs HF's own attention at the decoded position, at two lengths."""
    for ctx_len in CTX_LENGTHS:
        step = reference.full_step(ctx_len=ctx_len, device=DEV)
        loaded = reference.load_mixer("full_attention", step.layer)
        out, _key, _value = loaded.full_attention(step.hidden_new, *step.mixer_acts)

        want = reference.full_mixer_oracle(step)
        agrees_as_a_component(out, want)


def test_the_step_returns_the_cache_entry_to_append():
    """The step's returned key and value are this token's cache entry: appending
    them to the cache it was given reproduces the cache a context one token
    longer would have produced.

    Checked against a rebuilt cache rather than against the step's own inputs, so
    a step that returned its inputs unchanged would fail. The append itself is the
    decoder's own ``advance_state``, so the rule is stated once.
    """
    step = reference.full_step(device=DEV)
    loaded = reference.load_mixer("full_attention", step.layer)
    _out, key, value = loaded.full_attention(step.hidden_new, *step.mixer_acts)

    want_key, want_value = reference.appended_cache_oracle(step)
    grown_key, grown_value = advance_state(
        "full_attention", (step.k_cache, step.v_cache), (key, value)
    )

    assert tuple(grown_key.shape) == tuple(want_key.shape)
    # The cache each entry is appended to is the oracle's own, so this token's
    # entry is the only computed part and the one the bound follows.
    for grown, want in ((grown_key, want_key), (grown_value, want_value)):
        agrees_as_a_component(grown, want)


def test_only_the_leading_rotary_dims_carry_a_position():
    """``partial_rotary_factor`` is 0.25, and this measures that it is.

    Two positions, one tensor: the entries past ``rotary_dim`` must be bit-equal
    between them, and the entries before it must not be. A kernel that rotated
    the whole head would fail the first; one that rotated nothing would fail the
    second. Asserted against the position embedding rather than against the
    configuration field, so it is the behaviour that is checked.
    """
    shape = reference.CONFIG
    shape_rotary_dim = int(
        shape.head_dim * float(shape.rope_parameters["partial_rotary_factor"])
    )
    torch.manual_seed(3)
    x = torch.randn(
        1, 1, shape.num_attention_heads, shape.head_dim, device=DEV,
        dtype=reference.DTYPE,
    )
    loaded = reference.load_mixer(
        "full_attention", reference.hf_layer("full_attention", DEV)
    )

    turned = []
    for position in (7, 19):
        cos, sin = reference.rope_caches(position + 1, DEV)
        turned.append(
            loaded.partial_rope(
                x, cos[-1:], sin[-1:], torch.zeros(1, device=DEV, dtype=torch.int32)
            )
        )
    tail = [item[..., shape_rotary_dim:] for item in turned]
    head = [item[..., : shape_rotary_dim] for item in turned]

    assert torch.equal(tail[0], tail[1])
    untouched = x[..., shape_rotary_dim:]
    agrees_as_a_component(tail[0], untouched)
    assert (head[0] - head[1]).abs().max().item() > 1e-2


def test_the_output_gate_is_applied():
    """Half of ``q_proj``'s fan-out never reaches a score, and this measures that
    it reaches the output instead.

    The gate is a sigmoid, so it lies strictly between 0 and 1: an implementation
    that ignored it would be uniformly larger, and one that applied it twice
    uniformly smaller. Both are caught by running the same step against a second
    loading whose gate weights are neutralised -- if the answer did not move, the
    gate is not being read.
    """
    step = reference.full_step(device=DEV)
    loaded = reference.load_mixer("full_attention", step.layer)
    out, _key, _value = loaded.full_attention(step.hidden_new, *step.mixer_acts)

    shape = reference.CONFIG
    # w_qg is [1, hidden, 2 * heads * head_dim] with the gate interleaved per
    # head; zeroing the gate half sends every gate to sigmoid(0) = 1/2.
    gated = loaded.constants["w_qg"].clone().reshape(
        1, shape.hidden_size, shape.num_attention_heads, 2 * shape.head_dim
    )
    gated[..., shape.head_dim:] = 0.0
    neutralised = dataclasses.replace(
        loaded,
        constants={
            **loaded.constants,
            "w_qg": gated.reshape(1, shape.hidden_size, 2 * shape.num_attention_heads * shape.head_dim).contiguous(),
        },
    )
    ungated, _key, _value = neutralised.full_attention(step.hidden_new, *step.mixer_acts)

    with pytest.raises(AssertionError):
        agrees_as_a_component(out, ungated)
