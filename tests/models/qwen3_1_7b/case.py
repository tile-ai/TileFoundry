"""This model's corpus entry: what is selected from it, and how it is judged.

Stated in the package that owns the model rather than in one shared list, so
adding a model touches the model's own directory and the registry only names it.
"""

from __future__ import annotations

from tests.models.corpus import (
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tests.models.qwen3_1_7b.model import Qwen3_1_7B, config
from tests.models.qwen3_1_7b.reference import (
    CTX_LEN,
    decoder_step_inputs,
    decoder_step_oracle,
    run_decoder_step,
)

#: The context length the cache-reading functions are asked about at. A decode
#: kernel's cost is dominated by the cache it streams, so the length is stated
#: rather than minimised: analysing at the shortest context that type-checks
#: would report a cost profile no deployment has.
ANALYZED_AT = {"ctx_len": 1024}

CASE = ModelCase(
    id="qwen3_1_7b",
    prototype=Qwen3_1_7B,
    scope="layer0",
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
        problem_sizes=(f"decode/ctx_len={CTX_LEN}",),
    ),
    analyze=(
        FunctionCase(
            id="qwen3_1_7b/analyze/input_rms_norm", selector="input_rms_norm"
        ),
        FunctionCase(
            id="qwen3_1_7b/analyze/self_attention",
            selector="self_attention",
            dims=ANALYZED_AT,
        ),
        FunctionCase(id="qwen3_1_7b/analyze/mlp", selector="mlp"),
        FunctionCase(
            id="qwen3_1_7b/analyze/placed_mlp",
            selector="placed_mlp",
        ),
        FunctionCase(
            id="qwen3_1_7b/analyze/decoder_layer",
            selector="decoder_layer",
            dims=ANALYZED_AT,
        ),
    ),
    schedule=(
        FunctionCase(
            id="qwen3_1_7b/schedule/decoder_layer",
            selector="decoder_layer",
            topology="cta",
            dims=ANALYZED_AT,
        ),
    ),
    sized=(
        SizedCase(
            id="qwen3_1_7b/sized/decoder_layer",
            selector="decoder_layer",
            dims={"ctx_len": 1024},
            ceiling={"ctx_len": config.max_position_embeddings - 1},
        ),
    ),
)


#: What the registry collects from this package. This model is one Module, so
#: there is one case, and it names itself as its own model.
CASES = (CASE,)

__all__ = ["ANALYZED_AT", "CASE", "CASES"]
