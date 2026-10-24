"""This model's corpus entry: what is selected from it, and how it is judged.

Three cases name the KDA, MLA, and MoE definitions directly. Each carries the
execution context declared by its module, while the report groups all three
under one model.

The `reference` is spent on KDA, which is BLOCKED. That is deliberate: of the three
kernels it is the one that distinguishes this model, and a capability matrix that
recorded only the two with oracles would report this model as covered. MLA (both
forms) and the MoE are measured against Hugging Face in
`tests/integration/installed/analyzable/smoke_kimi_linear_48b_a3b.py`, at f32
round-off, each with perturbation tests establishing that those comparisons can
fail: two for MLA, and three for the MoE covering the router bias, the order the
routed scaling is applied in, and the shared expert's contribution.

Analyze selects everything the tree defines. What is not selected is untested,
and the report derives that from the model's own function inventory.
"""

from __future__ import annotations

from tests.nvidia.cuda.models.corpus import (
    CapabilityGate,
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tilefoundry_nvidia.cuda.models.kimi_linear_48b_a3b.model import KimiKda, KimiMla, KimiMoe, published
from tests.nvidia.cuda.models.kimi_linear_48b_a3b.reference import (
    CTX_LEN,
    KDA_BLOCK_REASON,
    kda_step_inputs,
    kda_step_oracle,
    run_kda_step,
)

#: The context length the cache-reading function is analysed at. Stated rather
#: than minimised: a decode kernel's cost is dominated by the cache it streams, so
#: analysing at the shortest context that type-checks would report a cost profile
#: no deployment has. Only MLA reads a context; the other two kernels carry no
#: range, so they state no extent.
ANALYZED_AT = {"ctx_len": 1024}

CASE = ModelCase(
    id="kimi_linear_48b_a3b",
    prototype=KimiKda,
    reference=ReferenceCase(
        id="kimi_linear_48b_a3b/reference/kda_decode",
        boundary=(
            "one decode step of a complete KDA layer -- the three short "
            "convolutions, the per-channel forget gate, the delta-rule state "
            "update and the gated output norm -- at production dimensions"
        ),
        inputs=kda_step_inputs,
        oracle=kda_step_oracle,
        # `runner`, not `entry`: the boundary is one Function, but the block has
        # to be raised from inside the gate, and `runner` is what the harness
        # calls there. See `run_kda_step`.
        runner=run_kda_step,
        problem_sizes=(f"decode/ctx_len={CTX_LEN}",),
        gate=CapabilityGate(outcome="BLOCKED", reason=KDA_BLOCK_REASON),
    ),
    analyze=(
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/kda_attention",
            selector="kda_attention",
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/short_conv", selector="short_conv"
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/l2_normalize", selector="l2_normalize"
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/kda_gate", selector="kda_gate"
        ),
    ),
)

MLA_CASE = ModelCase(
    id="kimi_linear_48b_a3b_mla",
    model="kimi_linear_48b_a3b",
    prototype=KimiMla,
    analyze=(
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/mla_attention",
            selector="mla_attention",
            dims=ANALYZED_AT,
        ),
    ),
    sized=(
        SizedCase(
            id="kimi_linear_48b_a3b/sized/mla_attention",
            selector="mla_attention",
            dims=ANALYZED_AT,
            ceiling={"ctx_len": published().model_max_length - 1},
        ),
    ),
)

MOE_CASE = ModelCase(
    id="kimi_linear_48b_a3b_moe",
    model="kimi_linear_48b_a3b",
    prototype=KimiMoe,
    analyze=(
        FunctionCase(id="kimi_linear_48b_a3b/analyze/moe", selector="moe"),
        FunctionCase(id="kimi_linear_48b_a3b/analyze/router", selector="router"),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/shared_expert",
            selector="shared_expert",
        ),
    ),
)

#: One case per selected definition, grouped under the same model.
CASES = (CASE, MLA_CASE, MOE_CASE)

__all__ = ["ANALYZED_AT", "CASE", "CASES", "MLA_CASE", "MOE_CASE"]
