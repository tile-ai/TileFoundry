"""What the complete-decoder Reference does not judge about one Qwen3-1.7B layer.

The corpus Reference runs the whole 28-layer decoder through the Evaluator and
compares it against Hugging Face, so the layer, its attention, its MLP and its norms
are all measured there -- through the same public entry a user comes in by, at
production dimensions, against a real oracle. Component tests of those would repeat
that comparison at less scope, which is why they are gone; `test_decoder.py` holds
the stack-level witness and `tests/models/test_reference_coverage.py` the corpus one.

What Reference genuinely cannot say, and what is left here: **the state the step
hands back.** A decode step returns the key and value its caller appends, and the
Reference compares only the value. A step that computed the right output and the
wrong cache entry would pass every comparison and then decode the next token from
a corrupted context.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.decode_oracle import agrees_to_one_rounding
from tests.models.qwen3_1_7b import reference

HIDDEN = reference.CONFIG.hidden_size

DEV = "cpu"


@pytest.mark.parametrize("ctx_len", [0, 24])
def test_decoder_layer_returns_the_cache_entry_to_append(ctx_len):
    """The step's returned key and value are this token's cache entry: appending
    them to the cache it was given reproduces the cache a context one token
    longer would have produced.

    Checked against a rebuilt cache rather than against the step's own inputs,
    so a step that returned its inputs unchanged would fail.

    A zero-length context is the first step of a sequence: nothing is cached, so
    the step attends the one token it brings itself and the cache it hands back is
    that token's single entry. The value the step computes is checked here too,
    because the corpus Reference runs at its own one context length and at zero
    nothing else says the step computed the right value.

    That value is checked twice: once at the layer's first real boundary -- the
    fused `input_layernorm + self_attn` ending at `o_proj` -- and once at the
    layer's own output. Splitting it that way localises a disagreement to one
    half or the other instead of reporting it against the composition.

    Both are held to one rounding, and both now have it to spare: the norms in
    this fixture are written out as `Qwen3RMSNorm` stages them rather than
    handed to the generic `tf.rms_norm`, so the halves compose without the
    cast-placement gap that used to sit between them.
    """
    drawn = reference.decode_step_inputs(ctx_len=ctx_len, device=DEV)
    out, k_new, v_new = drawn.loaded.decoder_layer(*drawn.args)

    # First half: attention through w_o.
    attention, _, _ = drawn.loaded.self_attention(*drawn.args)
    want_attention = reference.attention_reference(
        drawn.layer, drawn.hidden_ctx, drawn.hidden_new
    )
    agrees_to_one_rounding(attention, want_attention)

    # Second half, and so the whole layer: residual, the post-attention norm and
    # the MLP, against `Qwen3DecoderLayer.forward`'s own output.
    agrees_to_one_rounding(out, reference.decode_step_oracle(drawn))


    want_k, want_v = reference.appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert drawn.k_cache.shape[1] == drawn.ctx_len
    assert grown_k.shape[1] == drawn.ctx_len + 1
    assert tuple(grown_k.shape) == tuple(want_k.shape)
    # One rounding each: the projection and rotary this token's entry came from.
    agrees_to_one_rounding(grown_k, want_k)
    agrees_to_one_rounding(grown_v, want_v)
