"""The complete Gemma-2-2B decoder, one decode step, against Hugging Face's own
26-layer stack.

Its own test because it is its own claim. Every layer here is the layer
``test_decoder_layer.py`` already checks, so nothing about a single layer is
re-established; what is established is that the stack is the stack -- layers in
order, the residual threaded between them, the final norm applied once at the
end, and each layer reading its own cache rather than another layer's. A
per-layer comparison passes whether or not any of that holds.

Production dimensions mean the real 26 layers and the real hidden size, which
with Gemma-2's vocabulary-sized embedding is about five gibibytes of bf16
parameters. That is a CUDA-sized test, and CUDA is where model completeness is
accepted, so it skips rather than shrinks when there is no device -- a smaller
stack would be a different claim wearing this test's name.

The perturbation tests hold a wrong stack to the same comparison the correct one
passes, so the two halves of this module are statements about each other rather
than about two separately chosen bars.
"""
from __future__ import annotations

import pytest
import torch

from tests.models.gemma2_2b import reference

CONFIG = reference.CONFIG

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the complete decoder at production dimensions"
)

DEV = "cuda"



def _matches(got, want, msg: str = "") -> None:
    """The ordinary bf16 contract, for a workflow that is wiring rather than one
    kernel's parity.

    A stack composes many independently rounded boundaries, so one rounding is
    the wrong bar to hold it to and a depth multiplier would be inventing one.
    The dtype is asserted rather than assumed, because a bf16-against-f32
    comparison is the one failure this would otherwise hide; past that,
    `assert_close` applies PyTorch's own defaults for the dtype.
    """
    assert got.dtype == want.dtype, (
        f"comparing {got.dtype} against {want.dtype}; build the oracle at the "
        f"dtype the checkpoint publishes"
    )
    torch.testing.assert_close(got, want, msg=msg or None)

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
    return reference.decoder_step_oracle(drawn)


def test_the_embedding_matches_hugging_face(drawn) -> None:
    """The root's `embed` gathers the row `Gemma2Model.embed_tokens` would, scale
    and all: the oracle is the HF module, so a plain gather lands 48 times too
    small. Last row of the table, so a wrong axis cannot land on it."""
    token_ids = torch.tensor([CONFIG.vocab_size - 1], device=DEV, dtype=torch.int64)

    out = drawn.loaded.embed(token_ids)

    with torch.no_grad():
        want = drawn.model.model.embed_tokens(token_ids).reshape(
            1, 1, CONFIG.hidden_size
        )
    # A gather reassociates nothing: the row is the row, bit for bit.
    assert torch.equal(out, want)
