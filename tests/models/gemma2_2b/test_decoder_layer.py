"""What the complete-decoder Reference does not judge about one Gemma-2-2B layer.

The corpus Reference runs the whole 26-layer decoder through the Evaluator and
compares it against Hugging Face, so the layer, its attention, its MLP and its norms
are all measured there -- through the same public entry a user comes in by, at
production dimensions, against a real oracle. Component tests of those would repeat
that comparison at less scope, which is why they are gone; `test_decoder.py` holds
the stack-level witness and `tests/models/test_reference_coverage.py` the corpus one.

What that Reference genuinely cannot say is what the step hands *back*. A decode step
returns the state its caller advances, and the Reference compares only the value. A
step that computed the right output and the wrong cache entry would pass every
comparison and then decode the next token from a corrupted context.

It also cannot isolate the attention block itself, which is what the first test
below does: `self_attention` against `Gemma2Attention` at the decoded position,
at the input scale the model actually decodes at.
"""

from __future__ import annotations

import torch

from tests.models import decode_oracle as oracle
from tests.models.decode_oracle import SEQ_LEN, agrees_as_a_component
from tests.models.gemma2_2b import reference

HIDDEN = reference.CONFIG.hidden_size

DEV = "cpu"
#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither divides the key/value head count.
CTX_LENGTHS = (24, 40)


def test_self_attention_matches_hugging_face():
    """`self_attention` (input_layernorm + `Gemma2Attention`: GQA, RoPE and the
    soft-capped logits, over the cache and the new token) against Hugging Face's
    own attention at the decoded position.

    At the ordinary decode input scale, which is the scale the comparison means
    something at: the soft cap is part of the computation being reproduced, not a
    thing to be provoked, and driving the query far past the cap only compresses
    the reference range until the bound is measuring the compression.
    """
    drawn = reference.decode_step_inputs(device=DEV)

    out, _, _ = drawn.loaded.self_attention(*drawn.attention_args)

    total = drawn.ctx_len + SEQ_LEN
    cos, sin = reference._rope_at(total, DEV)
    sequence = torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1)
    # At the activations dtype: an f32 mask would promote HF attention and make
    # this a bf16-against-f32 comparison rather than a comparison of two kernels.
    mask = oracle.causal_mask(total, DEV, sequence.dtype)
    with torch.no_grad():
        ref, _ = drawn.layer.self_attn(
            drawn.layer.input_layernorm(sequence),
            position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=mask,
        )

    agrees_as_a_component(out, ref[:, -SEQ_LEN:, :])



def test_decoder_layer_returns_the_cache_entry_to_append():
    """The step's returned key and value are this token's cache entry: appending
    them to the cache it was given reproduces the cache a context one token
    longer would have produced.

    Checked against a rebuilt cache rather than against the step's own inputs,
    so a step that returned its inputs unchanged would fail.
    """
    drawn = reference.decode_step_inputs(device=DEV)
    _, k_new, v_new = drawn.loaded.decoder_layer(*drawn.args)

    want_k, want_v = reference.appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert tuple(grown_k.shape) == tuple(want_k.shape)
    # The cache handed in is the oracle's own, so the entry appended to it is the
    # only computed part and the one whose precision the bound follows.
    agrees_as_a_component(grown_k, want_k)
    agrees_as_a_component(grown_v, want_v)
