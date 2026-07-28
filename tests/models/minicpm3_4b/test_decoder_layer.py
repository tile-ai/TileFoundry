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

from tests.models.minicpm3_4b import config, reference
from tests.models.minicpm3_4b import minicpm3_4b as model
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.hir.specialize import specialize_concretely

HIDDEN = config.REAL.hidden
SEQ = config.SEQ_LEN

DEV = "cpu"
ATOL = RTOL = 2e-4

#: Two lengths, so a kernel that only works at the length it was authored
#: against cannot pass. Neither is a multiple of the head count.
CTX_LENGTHS = (24, 40)


def test_decoder_layer_returns_the_cache_entry_to_append():
    """The step's returned key and value are this token's cache entry: appending
    them to the cache it was given reproduces the cache a context one token
    longer would have produced.

    For MLA that is the claim the whole cache design rests on -- the entry is the
    assembled per-head key and the up-projected value, which is what Hugging
    Face's own cache holds (see ``config.py``). Checked against a rebuilt cache
    rather than against the step's own inputs, so a step that returned its inputs
    unchanged would fail.
    """
    drawn = reference.decode_step_inputs(device=DEV)
    fn = specialize_concretely(model.decoder_layer, {"ctx_len": drawn.ctx_len})
    _, k_new, v_new = evaluate(fn, *drawn.args, device=DEV)

    want_k, want_v = reference.appended_cache_oracle(drawn)
    grown_k = torch.cat([drawn.k_cache, k_new], dim=1)
    grown_v = torch.cat([drawn.v_cache, v_new], dim=1)

    assert tuple(grown_k.shape) == tuple(want_k.shape)
    assert tuple(grown_v.shape) == tuple(want_v.shape)
    torch.testing.assert_close(grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL)
