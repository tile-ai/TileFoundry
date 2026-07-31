"""Qwen3.5-35B-A3B's MoE block against Hugging Face's own.

The whole block, at the published 256 experts and top-8 -- not a smaller expert
count standing in for it. The router softmaxes over every expert before the top-8
is taken, so at 8 experts the surviving weights would be different numbers, and a
kernel that got those wrong would pass.

Both sides run at the dtype the checkpoint publishes and each comparison allows
one rounding at its reference's scale. The f32-era measured maximum
absolute difference is recorded in each test, so a regression that stays inside
the tolerance is still visible as a changed number.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.decode_oracle import agrees_as_a_component
from tests.models.qwen3_5_35b_a3b import reference

DEV = reference.DEVICE


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

    out = loaded(hidden)
    want = reference.moe_oracle(layer, hidden)

    agrees_as_a_component(out, want)


def test_routing_selects_the_experts_hugging_face_selects():
    """The router's choice, not just its arithmetic.

    Expert selection is an index, and an index that is wrong by one is not
    slightly wrong -- it runs a different expert. Checked as a set because
    nothing downstream depends on the order the eight arrive in: they are
    renormalised and summed.
    """
    layer, hidden, loaded = _step()
    tokens = layer.post_attention_layernorm(hidden).reshape(1, reference.CONFIG.hidden_size)

    got_weights, got_indices = loaded.router.routing(tokens)
    with torch.no_grad():
        _logits, want_weights, want_indices = layer.mlp.gate(tokens)

    assert set(got_indices.flatten().tolist()) == set(want_indices.flatten().tolist())
    assert got_indices.shape == (1, reference.CONFIG.num_experts_per_tok)
    order = torch.argsort(got_indices, dim=-1)
    want_order = torch.argsort(want_indices, dim=-1)
    got_sorted = got_weights.gather(-1, order)
    want_sorted = want_weights.gather(-1, want_order)
    agrees_as_a_component(got_sorted, want_sorted)


def test_the_shared_expert_is_part_of_the_block():
    """Dropping the shared expert changes the answer.

    ``shared_expert`` and ``shared_expert_gate`` appear in no published
    configuration field, so a fixture assembled from the configuration alone
    would omit them and every routed number would still be right. This measures
    that the omission would not go unnoticed -- if the shared contribution were
    negligible, the whole boundary would be weaker than it looks.
    """
    layer, hidden, loaded = _step()
    tokens = layer.post_attention_layernorm(hidden).reshape(1, reference.CONFIG.hidden_size)

    routed_weights, indices = loaded.router.routing(tokens)
    routed = loaded.routed_experts(tokens, routed_weights, indices)
    shared = loaded.shared_expert(tokens)
    want = reference.moe_oracle(layer, hidden)

    together = (routed + shared).reshape(want.shape)
    agrees_as_a_component(together, want)
    # Omitting the shared expert has to break the comparison the whole block
    # passes, or this test would hold for a fixture that never added it.
    with pytest.raises(AssertionError):
        agrees_as_a_component(routed.reshape(want.shape), want)
