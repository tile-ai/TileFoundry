"""Which models the corpus contains, and which of their functions are tested.

One list, read by the tests that run the cases and by the report that says what
ran. A model appears here once; what is not selected here is untested, and the
report derives that from the model's own function inventory rather than from a
second list somebody has to remember to update.

Every gate below states a limit that was measured, not one that was expected.
A gate is a claim about today: if the limit it names is lifted and the case
starts passing, the case fails until this list is corrected, so the matrix
cannot quietly drift into describing a system nobody has.

Analyze selects every function a model defines. Schedule cannot: the device-wide
partition algorithm decides the launch, so it admits only the module entry
function, and selecting a leaf for it would be selecting something the algorithm
has no answer to rather than something it answers badly.

`sized` is a third question, asked separately because a model can answer the
others without answering it: whether it can be analysed at a context length of
the caller's choosing. A model authored as one fixed shape analyses and schedules
perfectly well and has no context length to state, and the two facts must not be
collapsed -- a working analysis recorded as broken, or a missing capability
recorded as nothing at all. It stays its own row once a model answers both, so
there is somewhere to record the next model that answers only one.

A case that names `dims` is asking about the model at those extents. A model that
leaves a dimension open cannot be asked about without them at all -- counting
elements needs a number and a range is not one -- so for those the extents are
part of the question rather than a refinement of it.
"""

from __future__ import annotations

from tests.models.corpus import (
    MODELS_ROOT,
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tests.models.qwen3_1_7b.config import REAL as QWEN3_1_7B_SHAPE
from tests.models.qwen3_1_7b.reference import (
    CTX_LEN as QWEN3_1_7B_CTX_LEN,
)
from tests.models.qwen3_1_7b.reference import (
    decoder_step_inputs,
    decoder_step_oracle,
    run_decoder_step,
)

#: The context length the cache-reading functions are asked about at. A decode
#: kernel's cost is dominated by the cache it streams, so the length is stated
#: rather than minimised: analysing at the shortest context that type-checks
#: would report a cost profile no deployment has.
ANALYZED_AT = {"ctx_len": 1024}

QWEN3_1_7B = ModelCase(
    id="qwen3_1_7b",
    source=MODELS_ROOT / "qwen3_1_7b" / "model" / "decoder_layer.py",
    entry="Qwen3_1_7B",
    namespace={"config": QWEN3_1_7B_SHAPE},
    reference=ReferenceCase(
        id="qwen3_1_7b/reference/full_decoder_decode",
        boundary=(
            "one decode step of the complete decoder -- every layer in order, "
            "the residual threaded between them and the final norm closing the "
            "stack -- at production dimensions"
        ),
        inputs=decoder_step_inputs,
        oracle=decoder_step_oracle,
        runner=run_decoder_step,
        problem_sizes=(f"decode/ctx_len={QWEN3_1_7B_CTX_LEN}",),
    ),
    analyze=(
        FunctionCase(
            id="qwen3_1_7b/analyze/input_rms_norm", function="input_rms_norm"
        ),
        FunctionCase(
            id="qwen3_1_7b/analyze/self_attention",
            function="self_attention",
            dims=ANALYZED_AT,
        ),
        FunctionCase(id="qwen3_1_7b/analyze/mlp", function="mlp"),
        FunctionCase(id="qwen3_1_7b/analyze/tiled_mlp", function="tiled_mlp"),
        FunctionCase(
            id="qwen3_1_7b/analyze/decoder_layer",
            function="decoder_layer",
            dims=ANALYZED_AT,
        ),
    ),
    schedule=(
        FunctionCase(
            id="qwen3_1_7b/schedule/decoder_layer",
            function="decoder_layer",
            topology="cta",
            dims=ANALYZED_AT,
        ),
    ),
    sized=(
        SizedCase(
            id="qwen3_1_7b/sized/decoder_layer",
            function="decoder_layer",
            dims={"ctx_len": 1024},
        ),
    ),
)

CORPUS: tuple[ModelCase, ...] = (QWEN3_1_7B,)


def case(model_id: str) -> ModelCase:
    """The one model case called *model_id*."""
    for model in CORPUS:
        if model.id == model_id:
            return model
    known = ", ".join(model.id for model in CORPUS)
    raise KeyError(f"no model case {model_id!r} in the corpus; it holds {known}")


__all__ = ["CORPUS", "QWEN3_1_7B", "case"]
