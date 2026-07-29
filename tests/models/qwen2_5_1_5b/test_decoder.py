"""The complete Qwen2.5-1.5B decoder, one decode step, against Hugging Face's own
28-layer stack.

Its own test because it is its own claim. Every layer here is the layer
``test_decoder_layer.py`` already checks, so nothing about a single layer is
re-established; what is established is that the stack is the stack -- layers in
order, the residual threaded between them, the final norm applied once at the
end, and each layer reading its own cache rather than another layer's. A
per-layer comparison passes whether or not any of that holds.

Production dimensions mean the real 28 layers and the real hidden size, which is
6.6 GiB of f32 parameters. That is a CUDA-sized test, and CUDA is where model
completeness is accepted, so it skips rather than shrinks when there is no device
-- a smaller stack would be a different claim wearing this test's name.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from tests.models.qwen2_5_1_5b import config, reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the complete decoder at production dimensions"
)

DEV = "cuda"
ATOL = RTOL = 2e-3
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
    model = config.build_hf_decoder(seed=0, device=DEV)
    torch.manual_seed(1)
    drawn = torch.randn(1, ctx_len + 1, config.REAL.hidden, device=DEV) * 0.1
    hidden_ctx, hidden_new = drawn[:, :ctx_len], drawn[:, ctx_len:]
    caches = config.decoder_context_kv(model, hidden_ctx, device=DEV)

    cfg = config.build_hf_config()
    cos_cache, sin_cache = config.rope_caches(cfg, config.REAL.max_pos, device=DEV)
    pos_ids = torch.tensor([ctx_len], device=DEV, dtype=torch.int32)
    scale = torch.full(
        (1, 1, 1, 1), model.model.layers[0].self_attn.scaling, device=DEV
    )

    return model, reference.load_decoder(model), hidden_ctx, hidden_new, caches, (
        cos_cache, sin_cache, pos_ids, scale
    )


def test_the_embedding_matches_hugging_face(drawn) -> None:
    """The root's `embed` gathers the row `Qwen2Model.embed_tokens` would, at the
    table's last row so a wrong axis or a truncated table cannot land on it."""
    model, loaded, *_ = drawn
    token_ids = torch.tensor([config.REAL.vocab - 1], device=DEV, dtype=torch.int64)

    out = loaded.embed(token_ids)

    with torch.no_grad():
        want = model.model.embed_tokens(token_ids).reshape(1, 1, config.REAL.hidden)
    torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_the_complete_decoder_matches_hugging_face(drawn) -> None:
    """Every layer, in order, plus the final norm."""
    model, loaded, hidden_ctx, hidden_new, caches, (cos, sin, pos_ids, scale) = drawn

    out, entries = loaded.decode_hidden(hidden_new, cos, sin, pos_ids, scale, caches)

    want = config.decoder_decode_reference(model, hidden_ctx, hidden_new)
    assert len(entries) == config.REAL.n_layers
    torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_every_layer_returns_its_own_cache_entry(drawn) -> None:
    """Appending each layer's returned entry to the cache it was given reproduces
    the cache a context one token longer would have produced -- per layer.

    Checked for all 28 rather than one, because the failure this catches is a
    layer reading or writing a neighbour's cache, which no single layer's test
    can see.
    """
    model, loaded, hidden_ctx, hidden_new, caches, (cos, sin, pos_ids, scale) = drawn

    _out, entries = loaded.decode_hidden(hidden_new, cos, sin, pos_ids, scale, caches)

    grown = loaded.append_cache(caches, entries)
    want = config.decoder_context_kv(
        model, torch.cat([hidden_ctx, hidden_new], dim=1), device=DEV
    )
    for index, ((grown_k, grown_v), (want_k, want_v)) in enumerate(zip(grown, want)):
        torch.testing.assert_close(
            grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL, msg=f"layer {index} keys"
        )
        torch.testing.assert_close(
            grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL, msg=f"layer {index} values"
        )


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


@pytest.mark.parametrize("description", sorted(_STACK_ERRORS))
def test_a_stack_that_is_wrongly_ordered_is_caught(drawn, description) -> None:
    """The comparison has to be able to fail, and to fail for these reasons.

    Without this, the passing test above would read as evidence about layer
    order while being satisfied by any order at all -- the agreement it reports
    comes from 28 layers that were each already checked on their own.
    """
    model, loaded, hidden_ctx, hidden_new, caches, (cos, sin, pos_ids, scale) = drawn
    want = config.decoder_decode_reference(model, hidden_ctx, hidden_new)

    broken, broken_caches = _STACK_ERRORS[description](loaded, caches)
    out, _entries = broken.decode_hidden(
        hidden_new, cos, sin, pos_ids, scale, broken_caches
    )

    with pytest.raises(AssertionError):
        torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_a_wrong_final_norm_is_caught(drawn) -> None:
    """The norm that closes the stack is applied, and is the model's own."""
    model, loaded, hidden_ctx, hidden_new, caches, (cos, sin, pos_ids, scale) = drawn
    unit = dataclasses.replace(
        loaded,
        constants={**loaded.constants, "gamma_final": torch.ones_like(model.model.norm.weight)},
    )
    want = config.decoder_decode_reference(model, hidden_ctx, hidden_new)

    out, _entries = unit.decode_hidden(
        hidden_new, cos, sin, pos_ids, scale, caches
    )

    with pytest.raises(AssertionError):
        torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)
