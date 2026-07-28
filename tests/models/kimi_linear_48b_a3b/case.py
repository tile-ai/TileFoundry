"""This model's corpus entries: what is selected from it, and how it is judged.

Three Modules, three cases, one model. The submodules are separate execution
domains -- an HIR Function may only call a Function its own Module owns -- so they
cannot be one case, and every case names this package as its model so the report
stays one row.

The `reference` is spent on KDA, which is BLOCKED. That is deliberate: of the three
submodules it is the one that distinguishes this model, and a capability matrix
that recorded only the two submodules with oracles would report this model as
covered. MLA (both forms) and the MoE are measured in `test_mla.py` and
`test_moe.py`, at f32 round-off, each with perturbation tests establishing that
those comparisons can fail.

Schedule admits only a module entry function, so it selects the two attention
entries and not the leaves; analyze selects everything each Module defines. What
is not selected is untested, and the report derives that from the models' own
function inventories.
"""

from __future__ import annotations

from tests.models.corpus import (
    MODELS_ROOT,
    CapabilityGate,
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tests.models.kimi_linear_48b_a3b.config import REAL as SHAPE
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
#: no deployment has.
ANALYZED_AT = {"ctx_len": 1024}

CASE = ModelCase(
    id="kimi_linear_48b_a3b",
    source=MODELS_ROOT / "kimi_linear_48b_a3b" / "model" / "submodules.py",
    entry="KimiKda",
    namespace={"config": SHAPE},
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
            id="kimi_linear_48b_a3b/analyze/kda_attention", function="kda_attention"
        ),
        FunctionCase(id="kimi_linear_48b_a3b/analyze/short_conv", function="short_conv"),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/l2_normalize", function="l2_normalize"
        ),
        FunctionCase(id="kimi_linear_48b_a3b/analyze/kda_gate", function="kda_gate"),
    ),
    schedule=(
        FunctionCase(
            id="kimi_linear_48b_a3b/schedule/kda_attention",
            function="kda_attention",
            topology="cta",
        ),
    ),
    #: KDA carries no dynamic dimension at all, so there is no context length to
    #: ask it about. That is not a missing capability -- it is what a fixed-size
    #: recurrent state means -- and it must not be recorded as one, so `sized`
    #: is empty rather than holding a fabricated extent.
    sized=(),
)

#: The MLA submodule's corpus entry, separate because a `ModelCase` holds one
#: Module and MLA is its own execution domain. It names this package as its model,
#: so it joins the same report row rather than reading as a second model.
#:
#: No harness reference: its comparison against Hugging Face lives in `test_mla.py`,
#: which measures both published rotary conventions and the NoPE identity that makes
#: `DeepseekV3Attention` equal this layer. What this case contributes is the
#: submodule's analysis and schedule coverage and its function inventory.
MLA_CASE = ModelCase(
    id="kimi_linear_48b_a3b_mla",
    model="kimi_linear_48b_a3b",
    source=MODELS_ROOT / "kimi_linear_48b_a3b" / "model" / "submodules.py",
    entry="KimiMla",
    namespace={"config": SHAPE},
    analyze=(
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/mla_attention",
            function="mla_attention",
            dims=ANALYZED_AT,
        ),
    ),
    schedule=(
        FunctionCase(
            id="kimi_linear_48b_a3b/schedule/mla_attention",
            function="mla_attention",
            topology="cta",
            dims=ANALYZED_AT,
        ),
    ),
    sized=(
        SizedCase(
            id="kimi_linear_48b_a3b/sized/mla_attention",
            function="mla_attention",
            dims=ANALYZED_AT,
        ),
    ),
)

#: The MoE submodule's corpus entry, separate for the same reason. Its own
#: comparison is in `test_moe.py`, which draws the router bias nonzero -- at zero
#: an implementation that gathered the biased scores is indistinguishable from a
#: correct one, so a zero-bias fixture would be a degenerate test.
MOE_CASE = ModelCase(
    id="kimi_linear_48b_a3b_moe",
    model="kimi_linear_48b_a3b",
    source=MODELS_ROOT / "kimi_linear_48b_a3b" / "model" / "submodules.py",
    entry="KimiMoe",
    namespace={"config": SHAPE},
    analyze=(
        FunctionCase(id="kimi_linear_48b_a3b/analyze/moe", function="moe"),
        FunctionCase(id="kimi_linear_48b_a3b/analyze/router", function="router"),
        FunctionCase(
            id="kimi_linear_48b_a3b/analyze/shared_expert", function="shared_expert"
        ),
    ),
    schedule=(
        FunctionCase(
            id="kimi_linear_48b_a3b/schedule/moe", function="moe", topology="cta"
        ),
    ),
    sized=(),
)

#: What the registry collects from this package: three Modules, one model.
CASES = (CASE, MLA_CASE, MOE_CASE)

__all__ = ["ANALYZED_AT", "CASE", "CASES", "MLA_CASE", "MOE_CASE"]
