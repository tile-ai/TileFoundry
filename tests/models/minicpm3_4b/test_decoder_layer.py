"""What the complete-decoder Reference does not judge about one MiniCPM3-4B layer.

The corpus Reference runs the whole 62-layer decoder through the Evaluator and
compares it against Hugging Face, so the layer, its attention, its MLP and its norms
are all measured there -- through the same public entry a user comes in by, at
production dimensions, against a real oracle. Component tests of those would repeat
that comparison at less scope, which is why they are gone; `test_decoder.py` holds
the stack-level witness and `tests/models/test_reference_coverage.py` the corpus one.

What that Reference genuinely cannot say is what the step hands *back*. A decode step
returns the state its caller advances, and the Reference compares only the value. A
step that computed the right output and the wrong cache entry would pass every
comparison and then decode the next token from a corrupted context.
"""

from __future__ import annotations

import torch

from tests.models.decode_oracle import agrees_to_one_rounding
from tests.models.minicpm3_4b import reference
from tests.models.minicpm3_4b.model import _QK_HEAD_DIM, MiniCPM3_4B, config

HIDDEN = reference.CONFIG.hidden_size

DEV = "cpu"
#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither is a multiple of the head count.
CTX_LENGTHS = (24, 40)


def test_generation_inputs_match_the_drawn_position() -> None:
    """The root supplies its token and positional activations for a decode step."""
    step = 24
    source = torch.arange(step + 2, dtype=torch.int64, device=DEV)
    sentinel = object()

    token_ids, cos, sin, pos_ids, scale, residual_scale, caches = (
        MiniCPM3_4B.prepare_inputs_for_generation(source[: step + 1], step, sentinel, device=DEV)
    )
    want_cos, want_sin = reference._rope_at(step + 1, DEV)

    assert torch.equal(token_ids, source[step].reshape(1))
    assert torch.equal(cos[: step + 1], want_cos)
    assert torch.equal(sin[: step + 1], want_sin)
    assert cos.shape == sin.shape == (config.max_position_embeddings, reference.CONFIG.head_dim)
    assert torch.equal(pos_ids, torch.tensor([step], device=DEV, dtype=torch.int32))
    assert scale.shape == (1, 1, 1, 1)
    torch.testing.assert_close(scale, torch.full_like(scale, _QK_HEAD_DIM ** -0.5))
    assert residual_scale.shape == (1, 1, 1)
    assert caches is sentinel


def test_decoder_layer_returns_the_cache_entry_to_append():
    """The step's returned key and value are this token's cache entry: appending
    them to the cache it was given reproduces the cache a context one token
    longer would have produced.

    For MLA that is the claim the whole cache design rests on -- the entry is the
    assembled per-head key and the up-projected value, which is what Hugging
    Face's own cache holds (see ``reference.py``). Checked against a rebuilt cache
    rather than against the step's own inputs, so a step that returned its inputs
    unchanged would fail.
    """
    drawn = reference.decode_step_inputs(device=DEV)
    _out, k_new, v_new = drawn.loaded.decoder_layer(*drawn.args)

    want_k, want_v = reference.appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert tuple(grown_k.shape) == tuple(want_k.shape)
    assert tuple(grown_v.shape) == tuple(want_v.shape)
    # The cache handed in is the oracle's own, so the entry appended to it is the
    # only computed part and the one whose precision the bound follows.
    agrees_to_one_rounding(grown_k, want_k)
    agrees_to_one_rounding(grown_v, want_v)
