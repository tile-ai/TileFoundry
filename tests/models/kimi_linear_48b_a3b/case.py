"""This model's corpus entry: what is selected from it, and how it is judged.

One case, one model, one root. The three kernels are separate execution domains --
an HIR Function may only call a Function its own Module owns -- so each is its own
child Module, and a case names them by the path it reached them through rather
than by naming three models.

The `reference` is spent on KDA, which is BLOCKED. That is deliberate: of the three
kernels it is the one that distinguishes this model, and a capability matrix that
recorded only the two with oracles would report this model as covered. MLA (both
forms) and the MoE are measured in `test_mla.py` and `test_moe.py`, at f32
round-off, each with perturbation tests establishing that those comparisons can
fail.

Schedule admits one function per execution Module, so it selects each child's
entry and not its leaves; analyze selects everything the tree defines. What is not
selected is untested, and the report derives that from the model's own function
inventory.
"""

from __future__ import annotations

from tests.models.corpus import (
    CapabilityGate,
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tests.models.kimi_linear_48b_a3b.config import REAL as CONFIG
from tests.models.kimi_linear_48b_a3b.model import KimiLinear48BA3B
from tests.models.kimi_linear_48b_a3b.reference import (
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
    prototype=KimiLinear48BA3B,
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
            selector="kda.kda_attention",
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/short_conv", selector="kda.short_conv"
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/l2_normalize", selector="kda.l2_normalize"
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/kda_gate", selector="kda.kda_gate"
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/mla_attention",
            selector="mla.mla_attention",
            dims=ANALYZED_AT,
        ),
        FunctionCase(id="kimi_linear_48b_a3b/analyze/moe", selector="moe.moe"),
        FunctionCase(id="kimi_linear_48b_a3b/analyze/router", selector="moe.router"),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/shared_expert",
            selector="moe.shared_expert",
        ),
    ),
    schedule=(
        FunctionCase(
            id="kimi_linear_48b_a3b/schedule/kda_attention",
            selector="kda.kda_attention",
            topology="cta",
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/schedule/mla_attention",
            selector="mla.mla_attention",
            topology="cta",
            dims=ANALYZED_AT,
        ),
        FunctionCase(
            id="kimi_linear_48b_a3b/schedule/moe",
            selector="moe.moe",
            topology="cta",
        ),
    ),
    #: Only MLA leaves a dimension open. KDA's state is fixed-size and the MoE's
    #: expert count is a constant of the published model, so neither has a size to
    #: be asked at -- which is what those shapes mean, not a capability they lack.
    sized=(
        SizedCase(
            id="kimi_linear_48b_a3b/sized/mla_attention",
            selector="mla.mla_attention",
            dims=ANALYZED_AT,
            ceiling={"ctx_len": CONFIG.max_ctx - 1},
        ),
    ),
)

#: What the registry collects from this package: one Module, one case.
CASES = (CASE,)

__all__ = ["ANALYZED_AT", "CASE", "CASES"]
