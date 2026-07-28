"""The complete Qwen2.5-1.5B decoder, one decode step, against Hugging Face's own
28-layer stack.

Its own test because it is its own claim. Every layer here is the layer
``test_decoder_layer.py`` already checks, so nothing about a single layer is
re-established; what is established is that the stack is the stack -- layers in
order, the residual threaded between them, the final norm applied once at the
end, and each layer reading its own cache rather than another layer's. A
per-layer comparison passes whether or not any of that holds.

Production dimensions mean the real 28 layers and the real hidden size, which is
4.6 GiB of f32 parameters. That is a CUDA-sized test, and CUDA is where model
completeness is accepted, so it skips rather than shrinks when there is no device
-- a smaller stack would be a different claim wearing this test's name.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.qwen2_5_1_5b import config
from tests.models.qwen2_5_1_5b.decoder import build_decoder

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
    need a wrong stack reorder copies of the two lists.
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
    scale = torch.full((1, 1, 1, 1), model.layers[0].self_attn.scaling, device=DEV)

    weights = [
        (
            layer.input_layernorm.weight,
            config.linear_weight(layer.self_attn.q_proj),
            layer.self_attn.q_proj.bias,
            config.linear_weight(layer.self_attn.k_proj),
            layer.self_attn.k_proj.bias,
            config.linear_weight(layer.self_attn.v_proj),
            layer.self_attn.v_proj.bias,

            config.linear_weight(layer.self_attn.o_proj),
            layer.post_attention_layernorm.weight,
            config.linear_weight(layer.mlp.gate_proj),
            config.linear_weight(layer.mlp.up_proj),
            config.linear_weight(layer.mlp.down_proj),
        )
        for layer in model.layers
    ]
    return model, hidden_ctx, hidden_new, caches, weights, (
        cos_cache, sin_cache, pos_ids, scale
    )


def test_the_complete_decoder_matches_hugging_face(drawn) -> None:
    """Every layer, in order, plus the final norm."""
    model, hidden_ctx, hidden_new, caches, weights, (cos, sin, pos_ids, scale) = drawn
    decoder = build_decoder().bind_final_norm(model.norm.weight)

    out, entries = decoder.forward(hidden_new, cos, sin, pos_ids, scale, weights, caches)

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
    model, hidden_ctx, hidden_new, caches, weights, (cos, sin, pos_ids, scale) = drawn
    decoder = build_decoder().bind_final_norm(model.norm.weight)

    _out, entries = decoder.forward(hidden_new, cos, sin, pos_ids, scale, weights, caches)

    want = config.decoder_context_kv(
        model, torch.cat([hidden_ctx, hidden_new], dim=1), device=DEV
    )
    for index, ((k_new, v_new), (want_k, want_v)) in enumerate(zip(entries, want)):
        grown_k = torch.cat([caches[index][0], k_new], dim=1)
        grown_v = torch.cat([caches[index][1], v_new], dim=1)
        torch.testing.assert_close(
            grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL, msg=f"layer {index} keys"
        )
        torch.testing.assert_close(
            grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL, msg=f"layer {index} values"
        )


#: Ways the stack can be wrong that no single layer's test can see, each as a
#: transform of the correctly-ordered per-layer arguments.
_STACK_ERRORS = {
    "two adjacent layers swapped": lambda w, c: (w[:1] + w[2:3] + w[1:2] + w[3:], c),
    "two layers' caches swapped": lambda w, c: (w, c[:1] + c[2:3] + c[1:2] + c[3:]),
    "layer order reversed": lambda w, c: (list(reversed(w)), c),
}


@pytest.mark.parametrize("description", sorted(_STACK_ERRORS))
def test_a_stack_that_is_wrongly_ordered_is_caught(drawn, description) -> None:
    """The comparison has to be able to fail, and to fail for these reasons.

    Without this, the passing test above would read as evidence about layer
    order while being satisfied by any order at all -- the agreement it reports
    comes from 28 layers that were each already checked on their own.
    """
    model, hidden_ctx, hidden_new, caches, weights, (cos, sin, pos_ids, scale) = drawn
    decoder = build_decoder().bind_final_norm(model.norm.weight)
    want = config.decoder_decode_reference(model, hidden_ctx, hidden_new)

    broken_weights, broken_caches = _STACK_ERRORS[description](weights, caches)
    out, _entries = decoder.forward(
        hidden_new, cos, sin, pos_ids, scale, broken_weights, broken_caches
    )

    with pytest.raises(AssertionError):
        torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_a_wrong_final_norm_is_caught(drawn) -> None:
    """The norm that closes the stack is applied, and is the model's own."""
    model, hidden_ctx, hidden_new, caches, weights, (cos, sin, pos_ids, scale) = drawn
    decoder = build_decoder().bind_final_norm(torch.ones_like(model.norm.weight))
    want = config.decoder_decode_reference(model, hidden_ctx, hidden_new)

    out, _entries = decoder.forward(
        hidden_new, cos, sin, pos_ids, scale, weights, caches
    )

    with pytest.raises(AssertionError):
        torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)
