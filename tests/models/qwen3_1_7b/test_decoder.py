"""The complete Qwen3-1.7B decoder, one decode step, against Hugging Face's own
28-layer stack.

Its own test because it is its own claim. Every layer here is the layer
``test_decoder_layer.py`` already checks, so nothing about a single layer is
re-established; what is established is that the stack is the stack -- layers in
order, the residual threaded between them, the final norm applied once at the
end, and each layer reading its own cache rather than another layer's. A
per-layer comparison passes whether or not any of that holds.

Production dimensions mean the real 28 layers and the real hidden size, which is
3.8 GiB of bf16 parameters. That is a CUDA-sized test, and CUDA is where model
completeness is accepted, so it skips rather than shrinks when there is no device
-- a smaller stack would be a different claim wearing this test's name.

What is *not* here is a positive tensor comparison of the whole stack, or of the
28 cache entries. Both were composition: every layer's input already carries the
layers before it, so agreement had to be bought at a depth-scaled tolerance that
would have accepted anything. What they claimed is claimed better elsewhere --
`tests/integration/qwen3_1_7b_l3.py` decodes sixteen greedy tokens from the real
published checkpoint through the installed wheel and requires the token ids to be
*equal*, which is layer order, the residual thread, the final norm and sixteen
steps of real cache growth at once and with no tolerance at all. What is left
here is what that cannot say: that this comparison can still fail, and for these
reasons.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from tests.models.qwen3_1_7b import reference

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
    """The root's `embed` gathers the row `Qwen3Model.embed_tokens` would, at the
    table's last row so a wrong axis or a truncated table cannot land on it."""
    model, loaded, *_ = drawn
    token_ids = torch.tensor([CONFIG.vocab_size - 1], device=DEV, dtype=torch.int64)

    out = loaded.embed(token_ids)

    with torch.no_grad():
        want = model.model.embed_tokens(token_ids).reshape(1, 1, CONFIG.hidden_size)
    # A gather reassociates nothing: the row is the row, bit for bit.
    assert torch.equal(out, want)


#: Ways the stack can be wrong that no single layer's test can see. The weights
#: are bound, so a wrong stack is a wrong *loading*: its layers reordered, or the
#: caches handed to the right layers in the wrong order.
_STACK_ERRORS = {
    "two adjacent layers swapped": lambda m, c: (
        _reordered(m, (0, 2, 1, *range(3, len(m.modules)))), c
    ),
    "two layers' caches swapped": lambda m, c: (m, c[:1] + c[2:3] + c[1:2] + c[3:]),
    "layer order reversed": lambda m, c: (
        _reordered(m, tuple(reversed(range(len(m.modules))))), c
    ),
}


def _reordered(loaded, order):
    """*loaded* with its layers visited in *order* -- the loading perturbed, since
    that is where the weights now live."""
    return dataclasses.replace(
        loaded, modules=tuple(loaded.modules[index] for index in order)
    )
