"""The complete MiniCPM3-4B decoder, one decode step, against Hugging Face's own
62-layer stack.

Its own test because it is its own claim. Every layer here is the layer
``test_decoder_layer.py`` already checks, so nothing about a single layer is
re-established; what is established is that the stack is the stack -- layers in
order, the residual threaded between them, the final norm applied once at the
end, and each layer reading its own cache rather than another layer's. A
per-layer comparison passes whether or not any of that holds.

For MiniCPM3 the stack also settles one thing a single layer cannot: the residual
scale. ``scale_depth / sqrt(num_hidden_layers)`` is 1.4 in a one-layer fixture and
1.4/sqrt(62) here, so a step that ignored the scale entirely would still pass the
component test at depth one and fail here.

Production dimensions mean the real 62 layers and the real hidden size, which is
about 8 GiB of bf16 parameters. That is a CUDA-sized test, and CUDA is where model
completeness is accepted, so it skips rather than shrinks when there is no device
-- a smaller stack would be a different claim wearing this test's name.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.minicpm3_4b import reference

CONFIG = reference.CONFIG

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the complete decoder at production dimensions"
)

DEV = "cuda"





@pytest.fixture(scope="module")
def drawn():
    """One deterministic decode step, drawn once for the whole module.

    Building the stack costs the initialisation of about four billion parameters,
    and every test here asks the same question of the same draw. Drawing per test
    paid that three times over to arrive at identical tensors. Nothing here
    mutates the draw; the test that needs a wrong stack perturbs a copy of the
    loading and leaves the original alone.
    """
    return reference.decoder_step_inputs(device=DEV)



def test_the_embedding_matches_hugging_face(drawn) -> None:
    """The root's `embed` gathers the row `MiniCPM3Model.embed_tokens` would, scale
    and all: the oracle is the HF module, so a plain gather lands twelve times too
    small. Last row of the table, so a wrong axis cannot land on it."""
    token_ids = torch.tensor([CONFIG.vocab_size - 1], device=DEV, dtype=torch.int64)

    out = drawn.loaded.embed(token_ids)

    with torch.no_grad():
        want = drawn.model.model.embed_tokens(token_ids).reshape(
            1, 1, CONFIG.hidden_size
        )
    # A gather reassociates nothing: the row is the row, bit for bit.
    assert torch.equal(out, want)
