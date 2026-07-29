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
about 16 GiB of f32 parameters. That is a CUDA-sized test, and CUDA is where model
completeness is accepted, so it skips rather than shrinks when there is no device
-- a smaller stack would be a different claim wearing this test's name.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from tests.models.minicpm3_4b import config, reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the complete decoder at production dimensions"
)

DEV = "cuda"
ATOL = RTOL = 2e-3

#: How much wider than the tolerance a perturbation has to show up as. A
#: discrimination test that only just failed would be reporting the tolerance
#: rather than the perturbation, so a margin is demanded rather than a failure.
#: Ten times is not arbitrary: the five perturbations measure 0.038, 0.039, 0.21,
#: 0.29 and 3.76 against this ``atol`` of 0.002, so the closest of them clears the
#: tolerance nineteen-fold and this bound by a factor of two. The absolute
#: numbers are small because MiniCPM3's residual scale (1.4/sqrt(62) = 0.178)
#: damps every branch it admits -- the yardstick that makes them large is the
#: correct stack's own agreement, 2.7e-7, five orders of magnitude below.
MARGIN = 10 * ATOL


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


def _difference(out, want) -> float:
    """Max absolute disagreement, as one number a test can hold a margin to."""
    return (out.float() - want.float()).abs().max().item()


def test_the_embedding_matches_hugging_face(drawn) -> None:
    """The root's `embed` gathers the row `MiniCPM3Model.embed_tokens` would, scale
    and all: the oracle is the HF module, so a plain gather lands twelve times too
    small. Last row of the table, so a wrong axis cannot land on it."""
    token_ids = torch.tensor([config.REAL.vocab - 1], device=DEV, dtype=torch.int64)

    out = drawn.loaded.embed(token_ids)

    with torch.no_grad():
        want = drawn.model.model.embed_tokens(token_ids).reshape(
            1, 1, config.REAL.hidden
        )
    torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_the_complete_decoder_matches_hugging_face(drawn) -> None:
    """Every layer, in order, plus the final norm."""
    out, entries = reference.run_decoder_step(drawn)

    want = reference.decoder_step_oracle(drawn)
    assert len(entries) == config.REAL.n_layers
    torch.testing.assert_close(out.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_every_layer_returns_its_own_cache_entry(drawn) -> None:
    """Appending each layer's returned entry to the cache it was given reproduces
    the cache a context one token longer would have produced -- per layer.

    Checked for all 62 rather than one, because the failure this catches is a
    layer reading or writing a neighbour's cache, which no single layer's test can
    see. The appending is the root's own ``append_cache``, which is what a decode
    loop calls, so this observes that method rather than a copy of it.
    """
    _out, entries = reference.run_decoder_step(drawn)

    grown = drawn.loaded.append_cache(drawn.caches, entries)
    want = config.decoder_context_kv(
        drawn.model, torch.cat([drawn.hidden_ctx, drawn.hidden_new], dim=1), device=DEV
    )
    for index, ((grown_k, grown_v), (want_k, want_v)) in enumerate(zip(grown, want)):
        torch.testing.assert_close(
            grown_k.float(), want_k.float(), atol=ATOL, rtol=RTOL, msg=f"layer {index} keys"
        )
        torch.testing.assert_close(
            grown_v.float(), want_v.float(), atol=ATOL, rtol=RTOL, msg=f"layer {index} values"
        )


#: Ways the stack can be wrong that no single layer's test can see, each as a
#: transform of the loading, the per-layer caches and the residual scale. The
#: weights are bound, so the first four are a wrong *loading*: layers reordered,
#: caches handed to the right layers in the wrong order, or the entry that closes
#: the stack replaced.
#:
#: The last one is MiniCPM3's alone, and it stays an activation: a scale of one is
#: the plain residual add every other model in the corpus makes, so this is the
#: perturbation that says ``scale_depth`` is not decoration. It belongs at the
#: stack rather than at one layer because 1.4/sqrt(62) is only the model's own
#: value at the model's own depth.
_STACK_ERRORS = {
    "two adjacent layers swapped":
        lambda m, c, r: (_reordered(m, (0, 2, 1, *range(3, len(m.modules)))), c, r),
    "two layers' caches swapped":
        lambda m, c, r: (m, c[:1] + c[2:3] + c[1:2] + c[3:], r),
    "layer order reversed":
        lambda m, c, r: (_reordered(m, tuple(reversed(range(len(m.modules))))), c, r),
    "wrong final norm":
        lambda m, c, r: (_closed_by(m, torch.ones_like(m.constants["gamma_final"])), c, r),
    "residual scaling dropped":
        lambda m, c, r: (m, c, torch.ones_like(r)),
}


def _reordered(loaded, order):
    """*loaded* with its layers visited in *order* -- the loading perturbed, since
    that is where the weights now live."""
    return dataclasses.replace(
        loaded, modules=tuple(loaded.modules[index] for index in order)
    )


def _closed_by(loaded, gamma_final):
    """*loaded* with the norm that closes the stack read from *gamma_final*."""
    return dataclasses.replace(
        loaded, constants={**loaded.constants, "gamma_final": gamma_final}
    )


def test_a_stack_that_is_wrongly_assembled_is_caught(drawn) -> None:
    """The comparison has to be able to fail, and to fail for these reasons.

    Without this, the passing test above would read as evidence about layer order
    while being satisfied by any order at all -- the agreement it reports comes
    from 62 layers that were each already checked on their own. Held to a margin
    rather than to `pytest.raises`, so a perturbation that only just registered
    would be a failure here rather than a pass.

    One test walking every perturbation rather than one parametrised test each,
    because a parallel copy of this module's fixture is 16 GiB of parameters on
    the device and measured at 20-31 GiB resident with the oracle's forward on top
    of them. Six test functions over eight workers exhausted a 140 GiB card;
    three fit. Splitting this claim five ways would buy nothing and cost the
    ability to run it.
    """
    want = reference.decoder_step_oracle(drawn)
    measured = {}
    for description, perturb in sorted(_STACK_ERRORS.items()):
        broken, caches, residual_scale = perturb(
            drawn.loaded, drawn.caches, drawn.residual_scale
        )
        out, _entries = broken.decode_hidden(
            drawn.hidden_new, drawn.cos_cache, drawn.sin_cache, drawn.pos_ids,
            drawn.scale, residual_scale, caches,
        )
        measured[description] = _difference(out, want)

    too_small = {name: gap for name, gap in measured.items() if gap <= MARGIN}
    assert not too_small, f"invisible at atol={ATOL}: {too_small}; all: {measured}"
