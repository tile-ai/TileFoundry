"""This model's corpus entry: what is selected from it, and how it is judged.

Stated in the package that owns the model rather than in one shared list, so
adding a model touches the model's own directory and the registry only names it.

Every gate below names one measured limit, and it is the same one: MLA's split
points are uneven (``qk_nope_head_dim=64`` against ``qk_rope_head_dim=32``,
``kv_lora_rank=256`` against 32), this repo's ``Split`` op takes a count and only
makes equal parts, so the model states them with ``tf.slice`` -- and no analysis
has a cost evaluator for ``Slice``. Measured per (function, family): the two
Slice-free functions analyse under all four families; the two that carry a Slice
fail under all four with ``no cost evaluator registered for Slice``, and the
partition path fails on the same op.

One caveat about the schedule gate, which nothing in this package can fix: the
partition path raises ``PartitionProblemError``, which is a ``ValueError`` and not
a ``ScheduleError``, while ``test_schedule_coverage.py`` holds a blocked schedule
case to ``expect=ScheduleError``. So the gate below states the right reason and
would still be recorded as a plain failure rather than as the expected block. A
``Slice`` cost evaluator retires both problems at once; until one exists, adding
this case to ``registry.CORPUS`` needs that or a harness that sees the error the
partitioner actually raises.
"""

from __future__ import annotations

from tests.models.corpus import (
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tests.models.minicpm3_4b.model import MAX_CTX, MiniCPM3_4B
from tests.models.minicpm3_4b.reference import (
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
    id="minicpm3_4b",
    prototype=MiniCPM3_4B,
    reference=ReferenceCase(
        id="minicpm3_4b/reference/full_decoder_decode",
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
            id="minicpm3_4b/analyze/input_rms_norm", selector="input_rms_norm"
        ),
        FunctionCase(
            id="minicpm3_4b/analyze/mla_attention",
            selector="mla_attention",
            dims=ANALYZED_AT,
        ),
        FunctionCase(id="minicpm3_4b/analyze/mlp", selector="mlp"),
        FunctionCase(
            id="minicpm3_4b/analyze/decoder_layer",
            selector="decoder_layer",
            dims=ANALYZED_AT,
        ),
    ),
    schedule=(
        FunctionCase(
            id="minicpm3_4b/schedule/decoder_layer",
            selector="decoder_layer",
            topology="cta",
            dims=ANALYZED_AT,
        ),
    ),
    sized=(
        SizedCase(
            id="minicpm3_4b/sized/decoder_layer",
            selector="decoder_layer",
            dims=ANALYZED_AT,
            ceiling={"ctx_len": MAX_CTX - 1},
        ),
    ),
)


#: What the registry collects from this package. This model is one Module, so
#: there is one case, and it names itself as its own model.
CASES = (CASE,)

__all__ = ["ANALYZED_AT", "CASE", "CASES"]
