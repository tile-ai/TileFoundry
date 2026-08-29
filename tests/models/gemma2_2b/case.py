"""Gemma-2-2B's entry in the model corpus.

Lives beside the model rather than in ``registry.py`` so the description and the
description's subject move together: the functions selected here are the
functions ``model.py`` defines, and a kernel renamed in one file
without the other is a mismatch inside one package rather than a broken import
across two.

Analyse selects every function the model defines -- what is not selected here is
untested, and the report derives that from the model's own function inventory.
``sized`` is a second question: whether the model can be analysed at a context
length of the caller's choosing.

Every case below is declared as passing, and two of the analyse cases and the
sized case do not pass yet -- not for a reason in this package. ``Gelu`` has an
evaluation handler and no *cost* evaluator, so anything that costs ``mlp`` or
``decoder_layer`` stops at ``no cost evaluator registered for Gelu``: all four
analysis families and the sized question alike. It is one registration next to
``Sigmoid`` / ``Softplus`` / ``Tanh`` / ``ReLU`` in
``src/tilefoundry/visitor_registry/op_cost.py`` (``_elementwise``, as those are),
and it is deliberately not made here -- this package's boundary is
``tests/models/gemma2_2b/``.

They are not recorded as ``BLOCKED`` because a block would describe a limit of
this model, which this is not: it is one missing cost registration, shared by
every case that touches ``Gelu``.
"""
from __future__ import annotations

from tests.models.corpus import (
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tests.models.gemma2_2b.model import Gemma2_2B, config
from tests.models.gemma2_2b.reference import (
    CTX_LEN,
    decoder_step_inputs,
    decoder_step_oracle,
    run_decoder_step,
)

#: The context length the cache-reading functions are asked about at. A decode
#: kernel's cost is dominated by the cache it streams, so the length is stated
#: rather than minimised. Well inside ``config.sliding_window`` (Gemma-2's sliding
#: window), where a sliding layer and a full layer are the same computation.
ANALYZED_AT = {"ctx_len": 1024}

CASE = ModelCase(
    id="gemma2_2b",
    prototype=Gemma2_2B,
    scope="layer0",
    reference=ReferenceCase(
        id="gemma2_2b/reference/full_decoder_decode",
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
        FunctionCase(id="gemma2_2b/analyze/input_rms_norm", selector="input_rms_norm"),
        FunctionCase(
            id="gemma2_2b/analyze/self_attention",
            selector="self_attention",
            dims=ANALYZED_AT,
        ),
        FunctionCase(id="gemma2_2b/analyze/mlp", selector="mlp"),
        FunctionCase(
            id="gemma2_2b/analyze/decoder_layer",
            selector="decoder_layer",
            dims=ANALYZED_AT,
        ),
    ),
    sized=(
        SizedCase(
            id="gemma2_2b/sized/decoder_layer",
            selector="decoder_layer",
            dims=ANALYZED_AT,
            ceiling={"ctx_len": config.sliding_window - 1},
        ),
    ),
)

#: What the registry collects from this package. This model is one Module, so
#: there is one case, and it names itself as its own model.
CASES = (CASE,)

__all__ = ["ANALYZED_AT", "CASE", "CASES"]
