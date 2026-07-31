"""The complete Qwen2.5-1.5B decoder, one decode step, against Hugging Face's own
28-layer stack.

Its own test because it is its own claim. Every layer here is the layer
``test_decoder_layer.py`` already checks, so nothing about a single layer is
re-established; what is established is that the stack is the stack -- layers in
order, the residual threaded between them, the final norm applied once at the
end, and each layer reading its own cache rather than another layer's. A
per-layer comparison passes whether or not any of that holds.

Production dimensions mean the real 28 layers and the real hidden size, which is
3.3 GiB of bf16 parameters. That is a CUDA-sized test, and CUDA is where model
completeness is accepted, so it skips rather than shrinks when there is no device
-- a smaller stack would be a different claim wearing this test's name.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.qwen2_5_1_5b import reference

CONFIG = reference.CONFIG

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the complete decoder at production dimensions"
)

DEV = "cuda"
CTX_LEN = 24




@pytest.fixture(scope="module")
def drawn():
    """One deterministic decode step, drawn once for the whole module.

    Building the stack costs about twenty seconds -- 1.4 billion parameters
    initialised and moved to the device -- and every test here asks the same
    question of the same draw. Drawing per test paid that six times over to
    arrive at identical tensors. Nothing here mutates the draw; the tests that
    need a wrong stack perturb a copy of the loading.
    """
    return _draw()


def _draw(ctx_len=CTX_LEN):
    """One deterministic decode step over a *ctx_len*-token context."""
    model = reference.build_decoder(seed=0, device=DEV)
    torch.manual_seed(1)
    drawn = (torch.randn(1, ctx_len + 1, CONFIG.hidden_size, device=DEV) * 0.1).to(
        reference.DTYPE
    )
    hidden_ctx, hidden_new = drawn[:, :ctx_len], drawn[:, ctx_len:]
    caches = reference.decoder_context_kv(model, hidden_ctx, device=DEV)

    cos_cache, sin_cache = reference.rope_caches(DEV)
    pos_ids = torch.tensor([ctx_len], device=DEV, dtype=torch.int32)
    scale = torch.full(
        (1, 1, 1, 1), model.model.layers[0].self_attn.scaling, device=DEV,
        dtype=reference.DTYPE,
    )

    return model, reference.load_decoder(model), hidden_ctx, hidden_new, caches, (
        cos_cache, sin_cache, pos_ids, scale
    )


def test_the_embedding_matches_hugging_face(drawn) -> None:
    """The root's `embed` gathers the row `Qwen2Model.embed_tokens` would, at the
    table's last row so a wrong axis or a truncated table cannot land on it."""
    model, loaded, *_ = drawn
    token_ids = torch.tensor([CONFIG.vocab_size - 1], device=DEV, dtype=torch.int64)

    out = loaded.embed(token_ids)

    with torch.no_grad():
        want = model.model.embed_tokens(token_ids).reshape(1, 1, CONFIG.hidden_size)
    # A gather reassociates nothing: the row is the row, bit for bit.
    assert torch.equal(out, want)
