"""Qwen3.5-35B-A3B's MoE block against Hugging Face's own.

The whole block, at the published 256 experts and top-8 -- not a smaller expert
count standing in for it. The router softmaxes over every expert before the top-8
is taken, so at 8 experts the surviving weights would be different numbers, and a
kernel that got those wrong would pass.

``ATOL`` / ``RTOL`` are f32 round-off with headroom. The measured maximum
absolute difference is recorded in each test, so a regression that stays inside
the tolerance is still visible as a changed number.
"""
from __future__ import annotations

import torch

from tests.models.qwen3_5_35b_a3b import config, reference

DEV = reference.DEVICE
ATOL = RTOL = 2e-5

#: Measured on an H200, f32: 2.38e-06 against a reference of maximum magnitude
#: 7.16. Asserted as an upper bound rather than printed, so the boundary's
#: accuracy is a property the suite holds rather than something a reader has to
#: go and reproduce.
MEASURED_MAX_ABS_DIFF = 4e-06


def _step():
    """The linear-attention layer's own MoE block, and that block loaded.

    Both published layer types end in the same block, so this boundary asks for
    one of them rather than instantiating a second three-gigabyte one. It is the
    same object ``test_decoder_layer`` uses, so a worker running both pays once.

    The block's weights are read on the device the layer holds them on, which is
    where its activations are drawn, so the loading and the draw agree on one.
    """
    step = reference.linear_step(device=DEV, whole_layer=True)
    return step.layer, step.hidden_new, reference.load_moe(step.layer)


def test_moe_matches_hugging_face():
    """moe (post_attention_layernorm + `Qwen3_5MoeSparseMoeBlock`) vs HF."""
    layer, hidden, loaded = _step()

    out = loaded.moe(hidden)
    want = reference.moe_oracle(layer, hidden)

    difference = (out.float() - want.float()).abs().max().item()
    assert difference <= MEASURED_MAX_ABS_DIFF, difference
    torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_routing_selects_the_experts_hugging_face_selects():
    """The router's choice, not just its arithmetic.

    Expert selection is an index, and an index that is wrong by one is not
    slightly wrong -- it runs a different expert. Checked as a set because
    nothing downstream depends on the order the eight arrive in: they are
    renormalised and summed.
    """
    layer, hidden, loaded = _step()
    tokens = layer.post_attention_layernorm(hidden).reshape(1, config.REAL.hidden)

    got_weights, got_indices = loaded.routing(tokens)
    with torch.no_grad():
        _logits, want_weights, want_indices = layer.mlp.gate(tokens)

    assert set(got_indices.flatten().tolist()) == set(want_indices.flatten().tolist())
    assert got_indices.shape == (1, config.REAL.top_k)
    order = torch.argsort(got_indices, dim=-1)
    want_order = torch.argsort(want_indices, dim=-1)
    torch.testing.assert_close(
        got_weights.gather(-1, order).float(),
        want_weights.gather(-1, want_order).float(),
        atol=ATOL, rtol=RTOL,
    )


def test_the_shared_expert_is_part_of_the_block():
    """Dropping the shared expert changes the answer.

    ``shared_expert`` and ``shared_expert_gate`` appear in no published
    configuration field, so a fixture assembled from the configuration alone
    would omit them and every routed number would still be right. This measures
    that the omission would not go unnoticed -- if the shared contribution were
    negligible, the whole boundary would be weaker than it looks.
    """
    layer, hidden, loaded = _step()
    tokens = layer.post_attention_layernorm(hidden).reshape(1, config.REAL.hidden)

    routed_weights, indices = loaded.routing(tokens)
    routed = loaded.routed_experts(tokens, routed_weights, indices)
    shared = loaded.shared_expert(tokens)
    want = reference.moe_oracle(layer, hidden)

    torch.testing.assert_close(
        (routed + shared).reshape(want.shape).float(), want.float(),
        atol=ATOL, rtol=RTOL,
    )
    routed_only = (routed.reshape(want.shape).float() - want.float()).abs().max().item()
    assert routed_only > 100 * ATOL, (
        f"the shared expert moves the block's output by only {routed_only}, so "
        f"a fixture that omitted it would pass this package's tolerance"
    )
