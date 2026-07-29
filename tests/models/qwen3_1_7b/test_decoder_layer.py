"""What the complete-decoder Reference does not judge about one Qwen3-1.7B layer.

The corpus Reference runs the whole 28-layer decoder through the Evaluator and
compares it against Hugging Face, so the layer, its attention, its MLP and its norms
are all measured there -- through the same public entry a user comes in by, at
production dimensions, against a real oracle. Component tests of those would repeat
that comparison at less scope, which is why they are gone; `test_decoder.py` holds
the stack-level witness and `tests/models/test_reference_coverage.py` the corpus one.

Two things that Reference genuinely cannot say, and they are what is left here:

- **the state it hands back.** A decode step returns the key and value its caller
  appends, and the Reference compares only the value. A step that computed the right
  output and the wrong cache entry would pass every comparison and then decode the
  next token from a corrupted context.
- **that the tiled rewrite is the same program.** `tiled_mlp` exists to be the loop
  nest a tiled target wants; nothing about the decoder's output distinguishes it from
  `mlp`, because it is only ever reached when somebody selects it.
"""
from __future__ import annotations

import torch

from tests.models.qwen3_1_7b import config, reference

HIDDEN = config.REAL.hidden
SEQ = config.SEQ_LEN

DEV = "cpu"
ATOL = RTOL = 2e-4

#: Where the tiled comparison runs. Unlike the stack tests, this is a cost
#: choice and not a scope one -- the two rewrites are the same program on either
#: device -- so it falls back rather than skipping.
TILED_DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_tiled_mlp_matches_untiled_mlp():
    """tiled_mlp (the K-loop / column-block rewrite of `mlp`) against `mlp`
    itself on the same inputs: the loop tiling only reassociates the K
    reduction, so the two must agree to f32 round-off. Also checked against
    HF, so a bug shared by both rewrites cannot hide."""
    layer = config.build_hf_layer(seed=0, device=TILED_DEV)
    torch.manual_seed(1)
    x = torch.randn(1, SEQ, HIDDEN, device=TILED_DEV) * 0.1
    loaded = reference.load_layer(layer)

    with torch.no_grad():
        ref = layer.mlp(layer.post_attention_layernorm(x))
    untiled = loaded.mlp(x)
    tiled = loaded.tiled_mlp(x)

    torch.testing.assert_close(tiled.float(), untiled.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(tiled.float(), ref.float(), atol=ATOL, rtol=RTOL)


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
