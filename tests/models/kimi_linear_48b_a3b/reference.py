"""The executable semantics Kimi-Linear-48B-A3B's submodules are held to.

Inputs and oracle are one pair on purpose: the oracle is a Hugging Face module
with random weights, so a factory returning only tensors would leave a test free
to score them against a differently initialised module. What each `*_inputs`
returns therefore carries both the evaluator's arguments and the module they were
drawn from. Everything is seeded, so a disagreement is a disagreement about the
compiler rather than about which random draw each side got.

Two of the three submodules have a real oracle. One does not, and that is the
honest headline for this model -- see `KDA_BLOCK_REASON`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tests.models.kimi_linear_48b_a3b.config import (
    REAL,
    SEQ_LEN,
    KimiLinearConfig,
    build_mla_attention,
    build_mla_hf_config,
    identity_rope_caches,
    linear_weight,
    mla_context_kv,
    mla_decode_reference,
    rms_norm,
    rope_caches,
)

#: Seeds, named so a change to either is a visible change to the reference.
WEIGHT_SEED = 0
ACTIVATION_SEED = 1

#: The context length a case is drawn at unless it states another. Small enough
#: to keep the oracle's full-sequence forward cheap, and not a multiple of the
#: head count, so an index arithmetic error cannot coincide with a head boundary.
CTX_LEN = 24

#: Activation draws the MoE is checked over. The decode contract fixes the token
#: count at the literal 1, so a single call routes one token and exercises one
#: expert set; breadth over *which* experts get selected therefore comes from
#: redrawing rather than from batching, which would contradict the contract.
MOE_DRAWS = (1, 2, 3, 4)


# ── KDA: no oracle ───────────────────────────────────────────────────────────


class KdaReferenceUnavailable(RuntimeError):
    """Raised instead of returning inputs there is no oracle to score."""


#: Why the KDA reference is blocked, as measured on 2026-07-28.
#:
#: It is the *reference* that is blocked, not the model: `model.py`
#: describes `kda_attention` completely, and it analyses and schedules. What is
#: missing is anything to check its values against.
#:
#: `transformers` 5.14.1 has no `kimi_linear` implementation: `KimiLinearForCausalLM`
#: appears nowhere in the installed package, `kimi_linear` is absent from
#: `CONFIG_MAPPING`, and `AutoConfig.from_pretrained` on the pinned REAL fails
#: offline both ways -- `trust_remote_code=False` raises ValueError ("contains
#: custom code which must be executed"), `trust_remote_code=True` raises OSError
#: ("does not appear to have a file named configuration_kimi.py").
#:
#: The nearest installed relative is `Qwen3NextGatedDeltaNet`, and it computes a
#: different function: its forget gate is one scalar per head
#: (`g_t = g[:, :, i]`, so the state decays uniformly), while KDA's is a 128-wide
#: vector per head applied column-wise. Substituting it would score KDA against
#: a model that is not KDA.
#:
#: Hand-writing the reference from the REAL was considered and rejected. It
#: would compare this package's guess against this package's other guess, and the
#: REAL does not determine the answer: `mla_use_nope: true` alongside
#: `qk_rope_head_dim: 64` leaves the scaling denominator undetermined, and the
#: measured cost of guessing it wrong there is 22.5%. The same class of ambiguity
#: covers KDA's gate placement and normalisation order.
#:
#: What would lift this: an independent implementation of KDA that can be run.
#: vLLM 0.18.0 ships one (`model_executor/layers/kda.py` plus
#: `layers/fla/ops/kda.py`, Apache-2.0) and it is present on this machine but not
#: importable -- an orphaned python3.13 site-packages under a 3.12 interpreter.
#: Vendoring it is a policy decision for the repo owner, not something this
#: package should take on its own.
KDA_BLOCK_REASON = (
    "no runnable KDA implementation: transformers 5.14.1 has no kimi_linear, "
    "Qwen3NextGatedDeltaNet's forget gate is scalar-per-head rather than "
    "per-channel, and hand-writing one would score a guess against a guess"
)


@dataclass(frozen=True)
class KdaStepInputs:
    """One KDA decode step's arguments.

    Random rather than drawn from a model, because there is no model to draw from.
    They are enough to *run* the boundary, which is not the same as scoring it --
    that is exactly the gap `KDA_BLOCK_REASON` records.
    """

    args: tuple


def kda_step_inputs(*, device: str = "cpu", seed: int = WEIGHT_SEED) -> KdaStepInputs:
    """Arguments of the right shapes for one KDA decode step."""
    torch.manual_seed(seed)

    def drawn(*sizes, sigma=0.05):
        return torch.randn(*sizes, device=device) * sigma

    window = REAL.short_conv_kernel_size - 1
    return KdaStepInputs(
        args=(
            drawn(1, SEQ_LEN, REAL.hidden_size),
            torch.ones(REAL.hidden_size, device=device),
            drawn(1, REAL.hidden_size, REAL.kda_proj),
            drawn(1, REAL.hidden_size, REAL.kda_proj),
            drawn(1, REAL.hidden_size, REAL.kda_proj),
            drawn(REAL.short_conv_kernel_size, REAL.kda_proj),
            drawn(REAL.short_conv_kernel_size, REAL.kda_proj),
            drawn(REAL.short_conv_kernel_size, REAL.kda_proj),
            drawn(1, window, REAL.kda_proj),
            drawn(1, window, REAL.kda_proj),
            drawn(1, window, REAL.kda_proj),
            drawn(1, REAL.hidden_size, REAL.kda_head_dim),
            drawn(1, REAL.kda_head_dim, REAL.kda_proj),
            drawn(REAL.kda_proj),
            drawn(REAL.kda_num_heads),
            drawn(1, REAL.hidden_size, REAL.kda_num_heads),
            drawn(1, REAL.hidden_size, REAL.kda_head_dim),
            drawn(1, REAL.kda_head_dim, REAL.kda_proj),
            torch.ones(REAL.kda_head_dim, device=device),
            drawn(1, REAL.kda_proj, REAL.hidden_size),
            drawn(1, REAL.kda_num_heads, REAL.kda_head_dim, REAL.kda_head_dim),
            torch.full((1, 1, 1), REAL.kda_scaling, device=device),
        )
    )


def run_kda_step(inputs: KdaStepInputs):
    """Run the KDA boundary, then report that it cannot be scored.

    The run is real: the complete layer is evaluated at production dimensions and
    its results are checked to be finite, so the boundary is genuinely exercised
    rather than skipped. What cannot happen afterwards is the comparison, because
    there is no oracle -- so this raises `AssertionError` carrying
    `KDA_BLOCK_REASON`, which is what the capability gate holds the block to.

    Failing here rather than in `kda_step_inputs` is deliberate. The reference
    harness calls `inputs()` outside the gate, so a fixture that raised would be
    recorded as an error in the harness instead of as this model's stated limit.
    """
    from tests.models.kimi_linear_48b_a3b.model import KimiLinear48BA3B  # noqa: PLC0415
    from tilefoundry.evaluator import evaluate  # noqa: PLC0415

    # Evaluated on whichever device the arguments were drawn on, rather than the
    # evaluator's default: this boundary is small and CPU-sized, and inheriting a
    # default of "cuda" would make a blocked reference depend on a free GPU.
    device = inputs.args[0].device.type
    out, state, *windows = evaluate(KimiLinear48BA3B.kda.lookup("kda_attention"), *inputs.args, device=device)
    assert torch.isfinite(out).all(), "KDA produced non-finite output"
    assert torch.isfinite(state).all(), "KDA produced a non-finite state"
    for window in windows:
        assert torch.isfinite(window).all(), "KDA produced a non-finite conv window"

    raise AssertionError(KDA_BLOCK_REASON)


def kda_step_oracle(inputs):
    """There is none. Unreachable: `run_kda_step` raises before this is called."""
    raise KdaReferenceUnavailable(KDA_BLOCK_REASON)


# ── MLA ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MlaStepInputs:
    """One drawn MLA decode step, and the attention module behind it."""

    args: tuple
    attention: object
    ctx_len: int
    nope: bool
    hidden_ctx: torch.Tensor
    hidden_new: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    gamma_in: torch.Tensor


def mla_step_inputs(
    *, ctx_len: int = CTX_LEN, device: str = "cpu", nope: bool = True
) -> MlaStepInputs:
    """One deterministic MLA decode step over a *ctx_len*-token context.

    *nope* selects Kimi's own form. It is not a different kernel: the same
    `mla_attention` runs either way, and NoPE is expressed by handing it
    `cos = 1, sin = 0`. `test_mla.py` measures that this rotary is exactly the
    identity, which is what makes the substitution a fact rather than a hope.
    """
    attention = build_mla_attention(seed=WEIGHT_SEED, device=device)
    caches = identity_rope_caches if nope else rope_caches
    cos, sin = caches(REAL, device=device)

    torch.manual_seed(ACTIVATION_SEED)
    drawn = torch.randn(1, ctx_len + 1, REAL.hidden_size, device=device) * 0.1
    hidden_ctx, hidden_new = drawn[:, :ctx_len], drawn[:, ctx_len:]

    # The input RMSNorm belongs to the decoder layer, not to DeepseekV3Attention.
    # The HIR fuses it, so the oracle is fed exactly the states that norm
    # produces. `gamma_in` is drawn rather than set to ones for two reasons: ones
    # would leave a bug in the norm's weight application invisible, and -- because
    # RMSNorm is scale-invariant -- a norm the oracle does not also apply is
    # absorbed by MLA's latent norm and shows up only on the shared rope path.
    gamma_in = torch.randn(REAL.hidden_size, device=device) * 0.1 + 1.0
    normed_ctx = rms_norm(hidden_ctx, gamma_in, REAL)

    k_cache, v_cache = mla_context_kv(attention, normed_ctx, cos, sin, REAL)

    # The token being decoded sits immediately after the context.
    pos_ids = torch.tensor([ctx_len], device=device, dtype=torch.int32)
    scale = torch.full((1, 1, 1, 1), REAL.mla_scaling, device=device)

    return MlaStepInputs(
        args=(
            hidden_new,
            gamma_in,
            linear_weight(attention.q_proj),
            linear_weight(attention.kv_a_proj_with_mqa),
            attention.kv_a_layernorm.weight,
            linear_weight(attention.kv_b_proj),
            cos,
            sin,
            pos_ids,
            k_cache,
            v_cache,
            scale,
            linear_weight(attention.o_proj),
        ),
        attention=attention,
        ctx_len=ctx_len,
        nope=nope,
        hidden_ctx=hidden_ctx,
        hidden_new=hidden_new,
        k_cache=k_cache,
        v_cache=v_cache,
        cos=cos,
        sin=sin,
        gamma_in=gamma_in,
    )


def mla_step_oracle(inputs: MlaStepInputs) -> torch.Tensor:
    """What DeepseekV3Attention produces for the same drawn step.

    Fed the normed states, because the HIR's kernel fuses the input RMSNorm.
    """
    return mla_decode_reference(
        inputs.attention,
        rms_norm(inputs.hidden_ctx, inputs.gamma_in, REAL),
        rms_norm(inputs.hidden_new, inputs.gamma_in, REAL),
        inputs.cos,
        inputs.sin,
    )


def mla_appended_cache_oracle(inputs: MlaStepInputs):
    """The cache the step's caller should hold afterwards.

    Built the same way the input cache was, over the context with the decoded
    token appended: the step's returned key and value are correct exactly when
    appending them reproduces this.
    """
    return mla_context_kv(
        inputs.attention,
        rms_norm(
            torch.cat([inputs.hidden_ctx, inputs.hidden_new], dim=1),
            inputs.gamma_in,
            REAL,
        ),
        inputs.cos,
        inputs.sin,
        REAL,
    )


# ── MoE ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MoeInputs:
    """One drawn MoE call, and the Hugging Face MoE behind it."""

    args: tuple
    hf_moe: object
    hidden: torch.Tensor
    normed: torch.Tensor
    gamma_post: torch.Tensor
    act_seed: int


def build_hf_moe(
    seed: int = WEIGHT_SEED,
    device: str = "cuda",
    config: KimiLinearConfig | None = None,
    n_experts: int | None = None,
):
    """A `DeepseekV3MoE` at Kimi's numbers, with a NONZERO router bias.

    The nonzero `e_score_correction_bias` is load-bearing and must not be
    "simplified" to the zeros buffer the class defaults to. The router selects on
    `sigmoid(logits) + bias` but takes the routing weights from the *unbiased*
    scores, so at bias = 0 an implementation that gathered the biased scores is
    indistinguishable from a correct one. Measured: with the bias drawn nonzero
    the selected expert set changes for 16/16 tokens and gathering the biased
    scores instead moves the weights by 1.08e-01; at bias = 0 it moves them by
    exactly 0.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: PLC0415
        DeepseekV3MoE,
    )

    config = config or REAL
    n_experts = config.num_experts if n_experts is None else n_experts
    cfg = build_mla_hf_config(config)
    cfg.num_local_experts = n_experts
    cfg.num_experts_per_tok = config.num_experts_per_token
    cfg.n_shared_experts = config.num_shared_experts
    cfg.n_group = 1
    cfg.topk_group = 1
    cfg.norm_topk_prob = True
    cfg.routed_scaling_factor = config.routed_scaling_factor

    torch.manual_seed(seed)
    with torch.device(device):
        moe = DeepseekV3MoE(cfg)
    moe = moe.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(0.0, 0.02)
        moe.gate.e_score_correction_bias.normal_(0.0, 0.5)
    return moe


def moe_inputs(
    *, act_seed: int = ACTIVATION_SEED, device: str = "cuda", seed: int = WEIGHT_SEED,
    hf_moe=None, n_experts: int | None = None,
) -> MoeInputs:
    """One deterministic MoE call for one token.

    *hf_moe* may be passed in to reuse an already-built module: at 256 experts its
    weights are about 7 GB, so rebuilding it per draw dominates the test.
    """
    moe = (
        build_hf_moe(seed=seed, device=device, config=REAL, n_experts=n_experts)
        if hf_moe is None
        else hf_moe
    )

    # The post-attention RMSNorm belongs to the layer; the HIR fuses it, so the
    # oracle is fed exactly the states that norm produces. Drawn rather than ones:
    # the router reads the normed states directly, with no scale-invariant stage
    # downstream to absorb a mismatch.
    torch.manual_seed(seed + 7919)
    gamma_post = torch.randn(REAL.hidden_size, device=device) * 0.1 + 1.0

    torch.manual_seed(act_seed)
    hidden = torch.randn(1, SEQ_LEN, REAL.hidden_size, device=device) * 0.1
    normed = rms_norm(hidden, gamma_post, REAL)

    gate_up = moe.experts.gate_up_proj
    w_gate = gate_up[:, : REAL.moe_intermediate_size, :].contiguous()
    w_up = gate_up[:, REAL.moe_intermediate_size :, :].contiguous()
    w_down = moe.experts.down_proj.contiguous()

    shared = moe.shared_experts
    return MoeInputs(
        args=(
            hidden,
            gamma_post,
            moe.gate.weight.t().contiguous(),
            moe.gate.e_score_correction_bias,
            torch.full((1, 1), REAL.routed_scaling_factor, device=device),
            w_gate,
            w_up,
            w_down,
            linear_weight(shared.gate_proj),
            linear_weight(shared.up_proj),
            linear_weight(shared.down_proj),
        ),
        hf_moe=moe,
        hidden=hidden,
        normed=normed,
        gamma_post=gamma_post,
        act_seed=act_seed,
    )


def moe_oracle(inputs: MoeInputs) -> torch.Tensor:
    """What DeepseekV3MoE produces for the same drawn call."""
    with torch.no_grad():
        return inputs.hf_moe(inputs.normed)


__all__ = [
    "ACTIVATION_SEED",
    "CTX_LEN",
    "KDA_BLOCK_REASON",
    "MOE_DRAWS",
    "WEIGHT_SEED",
    "KdaReferenceUnavailable",
    "KdaStepInputs",
    "MlaStepInputs",
    "MoeInputs",
    "build_hf_moe",
    "kda_step_inputs",
    "kda_step_oracle",
    "mla_appended_cache_oracle",
    "mla_step_inputs",
    "mla_step_oracle",
    "moe_inputs",
    "moe_oracle",
    "run_kda_step",
]
