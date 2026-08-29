"""DeepSeek-V4-Flash as one corpus case: what runs, what is analysed, and at
what context length.

The boundary this model states is the sliding-window MLA attention submodule of
its first sliding layer, named as that -- ``DeepseekV4Flash.layer0.attention``,
the layer's own copy rather than the component it was built from, so what the
corpus reports is a Module the published model really holds. The tree above it --
embedding, 43 decoder layers, MoE, final norm, head -- is a real end-to-end path
with a real checkpoint pipeline behind it (`test_causal_lm_e2e.py`), and it is a
tree of Modules walked by orchestration methods rather than a Module of Functions:
a `ModelCase` names one Module and analysis selects Functions of that one
Module, so naming the root would put every kernel that does arithmetic
out of reach and report the model as three norms and an add.

`ctx_len` is bounded by the window rather than by the position embedding, so the
lengths here are small: a query in a sliding layer attends `window` positions
counting its own, and asking this description about the corpus's usual 1024 is
rejected outright rather than answered -- the envelope is `[1, 128)`, and a
context this layer type cannot attend has no cost profile worth reporting.
"""

from __future__ import annotations

from tests.models.corpus import (
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tests.models.deepseek_v4_flash.model import REAL, DeepseekV4Flash
from tests.models.deepseek_v4_flash.reference import (
    CTX_LEN,
    attention_step_inputs,
    attention_step_oracle,
    run_attention_step,
)

#: The context length the cache-reading function is asked about at. A decode
#: kernel's cost is dominated by the cache it streams, so the length is stated
#: rather than minimised -- but stated below the ceiling, because the largest
#: context is its own question (`config.DSV4Config.max_ctx`) and asking both at
#: one length would answer neither. Three quarters of the window, and not a
#: divisor of the head count, so an index error cannot land on a head boundary.
ANALYZED_AT = {"ctx_len": 96}

CASE = ModelCase(
    id="deepseek_v4_flash",
    prototype=DeepseekV4Flash,
    scope="layer0.attention",
    reference=ReferenceCase(
        id="deepseek_v4_flash/reference/attention_decode",
        boundary=(
            "one decode step of the complete sliding-window MLA attention "
            "submodule -- the fp8 KV latent this token writes and the "
            "online-softmax attention over the context it was given, with the "
            "weights bound by name the way the checkpoint binds them -- at "
            "production dimensions"
        ),
        inputs=attention_step_inputs,
        oracle=attention_step_oracle,
        runner=run_attention_step,
        problem_sizes=(f"decode/ctx_len={CTX_LEN}",),
    ),
    analyze=(
        FunctionCase(
            id="deepseek_v4_flash/analyze/mla_kv_update",
            selector="mla_kv_update",
        ),
        FunctionCase(
            id="deepseek_v4_flash/analyze/mla_attend",
            selector="mla_attend",
            dims=ANALYZED_AT,
        ),
    ),
    sized=(
        SizedCase(
            id="deepseek_v4_flash/sized/mla_attend",
            selector="mla_attend",
            dims=ANALYZED_AT,
            ceiling={"ctx_len": REAL.max_ctx},
        ),
    ),
)

#: What the registry collects from this package. This model is one Module, so
#: there is one case, and it names itself as its own model.
CASES = (CASE,)

__all__ = ["ANALYZED_AT", "CASE", "CASES"]
