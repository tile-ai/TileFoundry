"""The complete Gemma-2-2B decoder, one decode step, against Hugging Face's own
26-layer stack.

Its own test because it is its own claim. Every layer here is the layer
``test_decoder_layer.py`` already checks, so nothing about a single layer is
re-established; what is established is that the stack is the stack -- layers in
order, the residual threaded between them, the final norm applied once at the
end, and each layer reading its own cache rather than another layer's. A
per-layer comparison passes whether or not any of that holds.

Production dimensions mean the real 26 layers and the real hidden size, which
with Gemma-2's vocabulary-sized embedding is about ten gibibytes of f32
parameters. That is a CUDA-sized test, and CUDA is where model completeness is
accepted, so it skips rather than shrinks when there is no device -- a smaller
stack would be a different claim wearing this test's name.

The perturbation tests each measure how far the wrong stack lands from the
reference, not merely that it lands outside tolerance. A discrimination test that
only asserted "differs" would pass on a difference of one ulp, which is not the
difference it means to be reporting.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from tests.models.gemma2_2b import config, reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the complete decoder at production dimensions"
)

DEV = "cuda"
ATOL = RTOL = 2e-3

#: How much further than tolerance a perturbation has to land to count as caught.
#: A discrimination test is only evidence if the margin is not marginal. Measured:
#: the correct stack lands at 0.02x tolerance and the four perturbations at
#: 400x-1400x, so this bar separates them by more than an order of magnitude
#: either way rather than sitting just past the agreement.
MARGIN = 100.0


@pytest.fixture(scope="module")
def drawn():
    """One deterministic decode step, drawn once for the whole module.

    Building the stack costs tens of seconds -- 2.6 billion parameters
    initialised on the device -- and every test here asks the same question of
    the same draw. Drawing per test would pay that six times over to arrive at
    identical tensors. Nothing here mutates the draw; the tests that need a wrong
    stack perturb a copy of the loading.
    """
    return reference.decoder_step_inputs(device=DEV)


@pytest.fixture(scope="module")
def want(drawn):
    """Hugging Face's output for the drawn step, computed once."""
    return reference.decoder_step_oracle(drawn).float()


def test_the_embedding_matches_hugging_face(drawn) -> None:
    """The root's `embed` gathers the row `Gemma2Model.embed_tokens` would, scale
    and all: the oracle is the HF module, so a plain gather lands 48 times too
    small. Last row of the table, so a wrong axis cannot land on it."""
    token_ids = torch.tensor([config.REAL.vocab - 1], device=DEV, dtype=torch.int64)

    out = drawn.loaded.embed(token_ids)

    with torch.no_grad():
        want = drawn.model.model.embed_tokens(token_ids).reshape(
            1, 1, config.REAL.hidden
        )
    torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_the_complete_decoder_matches_hugging_face(drawn, want) -> None:
    """Every layer, in order, plus the final norm."""
    out, entries = drawn.loaded.decode_hidden(*drawn.args)

    assert len(entries) == config.REAL.n_layers
    torch.testing.assert_close(out.float(), want, atol=ATOL, rtol=RTOL)


def test_every_layer_returns_its_own_cache_entry(drawn) -> None:
    """Appending each layer's returned entry to the cache it was given reproduces
    the cache a context one token longer would have produced -- per layer.

    Checked for all 26 rather than one, because the failure this catches is a
    layer reading or writing a neighbour's cache, which no single layer's test
    can see. The appending is the root's own ``append_cache``, which is what a
    decode loop calls, so this observes that method rather than a copy of it.
    """
    _out, entries = drawn.loaded.decode_hidden(*drawn.args)

    grown = drawn.loaded.append_cache(drawn.caches, entries)
    grown_context = torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1)
    expected = config.decoder_context_kv(drawn.model, grown_context, device=DEV)
    for index, ((grown_k, grown_v), (want_k, want_v)) in enumerate(zip(grown, expected)):
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
def test_a_stack_that_is_wrongly_ordered_is_caught(drawn, want, description) -> None:
    """The comparison has to be able to fail, and to fail for these reasons.

    Without this, the passing test above would read as evidence about layer
    order while being satisfied by any order at all -- the agreement it reports
    comes from 26 layers that were each already checked on their own.
    """
    broken, broken_caches = _STACK_ERRORS[description](drawn.loaded, drawn.caches)
    out, _entries = broken.decode_hidden(
        drawn.hidden_new, drawn.cos_cache, drawn.sin_cache, drawn.pos_ids,
        drawn.scale, broken_caches,
    )

    _assert_far_from(out, want, description)


def test_a_wrong_final_norm_is_caught(drawn, want) -> None:
    """The norm that closes the stack is applied, and is the model's own weight
    adjusted the way ``Gemma2RMSNorm`` adjusts it.

    The wrong weight tried is the raw ``norm.weight`` -- ``config.rms_gamma``'s
    ``1.0 +`` forgotten. That is the mistake this model specifically invites, and
    it is also a different tensor from the right one, so it stands in for a norm
    that is not the model's own at all.
    """
    raw = dataclasses.replace(
        drawn.loaded,
        constants={**drawn.loaded.constants, "gamma_final": drawn.model.model.norm.weight},
    )

    out, _entries = raw.decode_hidden(*drawn.args)

    _assert_far_from(out, want, "wrong final norm")


def _assert_far_from(out, want, description) -> None:
    """*out* differs from *want* by far more than tolerance, and say by how much.

    The tolerance a passing comparison uses is ``atol + rtol * |want|``, so that
    is what the observed difference is measured against -- a perturbation is only
    evidence of discrimination if it clears the same bar the agreement does.
    """
    diff = (out.float() - want).abs()
    allowed = ATOL + RTOL * want.abs()
    ratio = (diff / allowed).max().item()
    assert ratio > MARGIN, (
        f"{description}: max difference {diff.max().item():.4g} is only "
        f"{ratio:.4g}x the tolerance -- too close to count as caught"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out.float(), want, atol=ATOL, rtol=RTOL)
