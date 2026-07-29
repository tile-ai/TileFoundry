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

Nor can it isolate the soft cap on the decoded token's own logit. That token is one
lane of one row, so a cap applied to the context and not to it moves the output by
very little -- little enough to pass a tolerance the rest of the layer needs.
"""

from __future__ import annotations

import torch

from tests.models import decode_oracle as oracle
from tests.models.gemma2_2b import config, reference

HIDDEN = config.REAL.hidden
SEQ = config.SEQ_LEN

DEV = "cpu"
ATOL = RTOL = 2e-4

#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither divides the key/value head count.
CTX_LENGTHS = (24, 40)

#: How much the drawn query is scaled up to make `attn_logit_softcapping` bite on
#: the new token's single logit rather than only on the cache's many.
SOFTCAP_PROBE_SCALE = 100.0


def test_self_attention_soft_caps_the_new_token_too():
    """The soft cap is on the new token's own logit as well as the cache's.

    A step that capped only the cache group would pass every other test in this
    file. The new token contributes one logit of `ctx_len + 1`, and at the drawn
    scale leaving it uncapped moves the output by 0.014x the tolerance -- below
    the round-off the comparison already allows. So the query is scaled until that
    single logit is far past the cap, where the same omission moves the output by
    9329x the tolerance instead, and the kernel is asked again.

    `scale` is the last of `self_attention`'s seven activations -- its projections
    are bound weights -- and x100 puts the raw logits at ~1900 against a cap of 50.
    """
    drawn = reference.decode_step_inputs(device=DEV)

    loud = list(drawn.attention_args)
    loud[-1] = loud[-1] * SOFTCAP_PROBE_SCALE
    out, _, _ = drawn.loaded.self_attention(*loud)

    cfg = config.build_hf_config()
    total = drawn.ctx_len + SEQ
    cos, sin = config.rope_caches(cfg, total, device=DEV)
    mask = oracle.causal_mask(total, DEV)
    sequence = torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1)
    attention = drawn.layer.self_attn
    with torch.no_grad():
        normed = drawn.layer.input_layernorm(sequence)
        held = attention.scaling
        attention.scaling = held * SOFTCAP_PROBE_SCALE
        try:
            ref, _ = attention(
                normed,
                position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
                attention_mask=mask,
            )
        finally:
            attention.scaling = held

    torch.testing.assert_close(
        out.float(), ref[:, -SEQ:, :].float(), atol=ATOL, rtol=RTOL
    )


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
    torch.testing.assert_close(grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL)
