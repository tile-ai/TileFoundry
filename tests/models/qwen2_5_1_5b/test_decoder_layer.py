"""What the complete-decoder Reference does not judge about one Qwen2.5-1.5B layer.

The corpus Reference runs the whole 28-layer decoder through the Evaluator and
compares it against Hugging Face, so the layer, its attention, its MLP and its norms
are all measured there -- through the same public entry a user comes in by, at
production dimensions, against a real oracle. Component tests of those would repeat
that comparison at less scope, which is why they are gone; `test_decoder.py` holds
the stack-level witness and `tests/models/test_reference_coverage.py` the corpus one.

What that Reference genuinely cannot say is what the step hands *back*. A decode step
returns the state its caller advances, and the Reference compares only the value. A
step that computed the right output and the wrong cache entry would pass every
comparison and then decode the next token from a corrupted context.

It also cannot say that the tiled rewrite is the same program: `tiled_mlp` is the loop
nest a tiled target wants, and nothing about the decoder's output distinguishes it
from `mlp`, because it is only ever reached when somebody selects it.
"""

from __future__ import annotations

import torch

from tests.models.qwen2_5_1_5b import config, reference
from tests.models.qwen2_5_1_5b import qwen2_5_1_5b as model
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.specialize import specialize_concretely

HIDDEN = config.REAL.hidden
SEQ = config.SEQ_LEN

DEV = "cpu"
ATOL = RTOL = 2e-4

#: Where the tiled comparison runs. Unlike the stack tests, this is a cost
#: choice and not a scope one -- the two rewrites are the same program on either
#: device -- so it falls back rather than skipping.
TILED_DEV = "cuda" if torch.cuda.is_available() else "cpu"

#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither divides the key/value head count.
CTX_LENGTHS = (24, 40)


def _one_token(device, seed=1):
    """A fresh HF layer and one token's hidden states."""
    layer = config.build_hf_layer(seed=0, device=device)
    torch.manual_seed(seed)
    return layer, torch.randn(1, SEQ, HIDDEN, device=device) * 0.1


def test_tiled_mlp_matches_untiled_mlp():
    """tiled_mlp (the K-loop / column-block rewrite of `mlp`) against `mlp`
    itself on the same inputs: the loop tiling only reassociates the K
    reduction, so the two must agree to f32 round-off. Also checked against
    HF, so a bug shared by both rewrites cannot hide."""
    layer, x = _one_token(TILED_DEV)
    mlp = layer.mlp
    weights = (
        layer.post_attention_layernorm.weight,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
    )

    with torch.no_grad():
        ref = mlp(layer.post_attention_layernorm(x))
    untiled = evaluate(model.mlp, x, *weights, device=TILED_DEV)
    tiled = evaluate(model.tiled_mlp, x, *weights, device=TILED_DEV)

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
    fn = specialize_concretely(model.decoder_layer, {"ctx_len": drawn.ctx_len})
    _, k_new, v_new = evaluate(fn, *drawn.args, device=DEV)

    want_k, want_v = reference.appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert tuple(grown_k.shape) == tuple(want_k.shape)
    torch.testing.assert_close(grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL)
