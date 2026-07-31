"""Qwen3.5-35B-A3B as corpus cases: what runs, what is analysed, what is
scheduled, and at what context length.

Three Modules, three cases. The published stack is a hybrid -- three linear
attention layers to every full attention one, each ending in the same 256-expert
MoE block -- and the three are different kernels, not one kernel configured three
ways. A `ModelCase` names one Module, so a single case would have to pick one of
them and the other two would be reported as nothing at all.

The boundary each case states is a complete submodule, not the causal LM. That is
the plan's rule for a model this size: the published model is 40 layers of 35
billion parameters, one whole decoder layer is 3.3 GB in f32, and a 40-layer walk
is neither affordable nor more informative than the layer it repeats. What a stack
would observe and these do not -- layer order, the residual thread, the final norm
-- is recorded as untested in `test_provenance.py` rather than approximated.

Two of the three carry a reference. The MoE block does not, for a measured reason
stated at its case; it is compared against Hugging Face in `test_moe.py`.

Two things this deliberately does not state at all:

- **No decoder-layer case.** Each published layer type is its own class holding
  its own mixer, so there are two of them and no single case names the layer.
  Their composition -- the two residuals, and the MoE reading the mixer's output --
  is measured in `test_decoder_layer.py` across both types.
- **No multi-token-prediction case.** `mtp_num_hidden_layers` is 1 and the
  installed transformers implements no head for it, so there is no oracle; a
  reference would be this repository's reading of a config compared against this
  repository's kernels. Measured and stated in `test_provenance.py`.
"""

from __future__ import annotations

from tests.models.corpus import (
    FunctionCase,
    ModelCase,
    ReferenceCase,
    SizedCase,
)
from tests.models.qwen3_5_35b_a3b.model import (
    MAX_CTX,
    Qwen3_5Decoder,
    config,
)
from tests.models.qwen3_5_35b_a3b.reference import (
    CTX_LEN,
    full_mixer_oracle,
    full_step,
    linear_mixer_oracle,
    linear_step,
    run_full_attention_step,
    run_linear_attention_step,
)

#: The context length the cache-reading function is asked about at. A decode
#: kernel's cost is dominated by the cache it streams, so the length is stated
#: rather than minimised -- but stated below `config.Qwen35Shape.max_ctx`, because
#: the largest context is its own question and asking both at one length would
#: answer neither.
ANALYZED_AT = {"ctx_len": 1024}

_LINEAR_LAYER = config.layer_types.index("linear_attention")
_FULL_LAYER = config.layer_types.index("full_attention")

#: The linear-attention layer is the model's own: three layers in four are one, and
#: the Gated DeltaNet is the semantics no other model in the corpus has. So it is
#: the case that carries the package's name.
CASE = ModelCase(
    id="qwen3_5_35b_a3b",
    prototype=Qwen3_5Decoder,
    scope=f"layer{_LINEAR_LAYER}.mixer",
    reference=ReferenceCase(
        id="qwen3_5_35b_a3b/reference/linear_attention_decode",
        boundary=(
            "one decode step of the complete Gated DeltaNet token mixer -- the "
            "causal short convolution over the state's own window, the "
            "L2-normalised query and key, the per-channel forget gate, the "
            "delta-rule state update and the gated output norm -- at production "
            "dimensions"
        ),
        inputs=linear_step,
        oracle=linear_mixer_oracle,
        runner=run_linear_attention_step,
        problem_sizes=(f"decode/ctx_len={CTX_LEN}",),
    ),
    analyze=(
        FunctionCase(
            id="qwen3_5_35b_a3b/analyze/linear_attention", selector="linear_attention"
        ),
        FunctionCase(id="qwen3_5_35b_a3b/analyze/conv_step", selector="conv_step"),
        FunctionCase(id="qwen3_5_35b_a3b/analyze/delta_step", selector="delta_step"),
        FunctionCase(
            id="qwen3_5_35b_a3b/analyze/l2_normalise", selector="l2_normalise"
        ),
    ),
    schedule=(
        FunctionCase(
            id="qwen3_5_35b_a3b/schedule/linear_attention",
            selector="linear_attention",
            topology="cta",
        ),
    ),
    #: The Gated DeltaNet's state is fixed-size, so no extent in this Module is
    #: left open and there is no context length to ask it about. That is what a
    #: recurrent state means rather than a missing capability, so `sized` is empty
    #: rather than holding a fabricated extent.
    sized=(),
)

#: One layer in four. The only case here whose kernel carries a range, because it
#: is the only one that reads a context rather than a state.
FULL_ATTENTION_CASE = ModelCase(
    id="qwen3_5_35b_a3b_full_attention",
    model="qwen3_5_35b_a3b",
    prototype=Qwen3_5Decoder,
    scope=f"layer{_FULL_LAYER}.mixer",
    reference=ReferenceCase(
        id="qwen3_5_35b_a3b/reference/full_attention_decode",
        boundary=(
            "one decode step of the complete full-attention token mixer -- GQA "
            "over the context given as tensors, per-head query and key norms, "
            "partial rotary embedding and the sigmoid output gate -- at "
            "production dimensions"
        ),
        inputs=full_step,
        oracle=full_mixer_oracle,
        # `runner`, not `entry`: the weights are bound by a loading, so what runs
        # the boundary is that loading rather than a Function taken positionally.
        runner=run_full_attention_step,
        problem_sizes=(f"decode/ctx_len={CTX_LEN}",),
    ),
    analyze=(
        FunctionCase(
            id="qwen3_5_35b_a3b/analyze/full_attention",
            selector="full_attention",
            dims=ANALYZED_AT,
        ),
        FunctionCase(
            id="qwen3_5_35b_a3b/analyze/partial_rope", selector="partial_rope"
        ),
        FunctionCase(
            id="qwen3_5_35b_a3b/analyze/partial_rope_kv", selector="partial_rope_kv"
        ),
    ),
    schedule=(
        FunctionCase(
            id="qwen3_5_35b_a3b/schedule/full_attention",
            selector="full_attention",
            topology="cta",
            dims=ANALYZED_AT,
        ),
    ),
    sized=(
        SizedCase(
            id="qwen3_5_35b_a3b/sized/full_attention",
            selector="full_attention",
            dims=ANALYZED_AT,
            ceiling={"ctx_len": MAX_CTX - 1},
        ),
    ),
)

#: Every layer ends in this block, at the published 256 experts and top-8 rather
#: than a smaller count standing in for it: the router softmaxes over every expert
#: before the top-8 is taken, so at 8 experts the surviving weights would be
#: different numbers and a kernel that got them wrong would pass.
MOE_CASE = ModelCase(
    id="qwen3_5_35b_a3b_moe",
    model="qwen3_5_35b_a3b",
    prototype=Qwen3_5Decoder,
    scope=f"layer{_LINEAR_LAYER}.moe",
    #: No harness reference. Drawing one means building a whole decoder layer for
    #: its block -- 805 million parameters, 3.3 GB in f32, essentially all of it the
    #: 256 experts -- which is by a wide margin the most expensive draw in this
    #: package, and the harness draws inputs for every model in one place. The block
    #: is compared against `Qwen3_5MoeSparseMoeBlock` in `test_moe.py` instead, with
    #: perturbation tests establishing that the comparison can fail; what this case
    #: contributes here is the block's analysis and schedule coverage and its
    #: function inventory.
    analyze=(
        FunctionCase(id="qwen3_5_35b_a3b/analyze/experts", selector="experts"),
        FunctionCase(id="qwen3_5_35b_a3b/analyze/post_norm", selector="post_norm"),
        FunctionCase(
            id="qwen3_5_35b_a3b/analyze/routing", selector="router.routing"
        ),
        FunctionCase(
            id="qwen3_5_35b_a3b/analyze/routed_experts", selector="routed_experts"
        ),
        FunctionCase(
            id="qwen3_5_35b_a3b/analyze/shared_expert", selector="shared_expert"
        ),
    ),
    schedule=(
        FunctionCase(
            id="qwen3_5_35b_a3b/schedule/experts", selector="experts", topology="cta"
        ),
    ),
    #: One token through a router; nothing here is authored over a context.
    sized=(),
)

#: What the registry collects from this package.
CASES = (CASE, FULL_ATTENTION_CASE, MOE_CASE)

__all__ = ["ANALYZED_AT", "CASE", "CASES", "FULL_ATTENTION_CASE", "MOE_CASE"]
