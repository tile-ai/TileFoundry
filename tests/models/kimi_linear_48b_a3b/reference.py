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

from tests.models import decode_oracle as oracle
from tests.models.decode_oracle import SEQ_LEN, linear_weight
from tests.models.kimi_linear_48b_a3b.model import (
    KimiLinearConfig,
    published,
)

#: The checkpoint's own configuration, read by the class it names.
CONFIG = published()


#: The precision the checkpoint publishes; both sides of every comparison run at it.
DTYPE = CONFIG.dtype

#: The same config with a quarter of the experts: the router claim is identical
#: at 64 and the full 256 will not fit alongside a parallel suite.
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


SMALL_MOE = published()
SMALL_MOE.num_experts = 64

# KDA's dimensions are published under `linear_attn_config`, so they carry the
# prefix: the top-level `head_dim: 72` belongs to neither path, and KDA's 128 and
# MLA's 192/128 are different numbers.
_KDA = CONFIG.linear_attn_config
KDA_HEAD_DIM = _KDA["head_dim"]
KDA_NUM_HEADS = _KDA["num_heads"]
SHORT_CONV_KERNEL_SIZE = _KDA["short_conv_kernel_size"]
KDA_PROJ = KDA_NUM_HEADS * KDA_HEAD_DIM

#: The scaling denominator is `qk_head_dim` (128 + 64), not `v_head_dim`; the
#: config alone does not say which, and vLLM uses `qk_head_dim ** -0.5`.
QK_HEAD_DIM = CONFIG.qk_nope_head_dim + CONFIG.qk_rope_head_dim
MLA_SCALING = QK_HEAD_DIM ** -0.5

#: `kda_head_dim ** -0.5`, applied to q *after* its l2 normalisation.
KDA_SCALING = KDA_HEAD_DIM ** -0.5



# ── the MLA oracle: DeepseekV3Attention at Kimi's ranks ───────────────────────


def build_mla_hf_config(config: KimiLinearConfig = CONFIG):
    """A `DeepseekV3Config` whose MLA is structurally Kimi's.

    Not a claim that Kimi is DeepSeek-V3. It is a claim about one submodule:
    at these ranks `DeepseekV3Attention` builds exactly the parameter set vLLM's
    `KimiMLAAttention` builds -- `q_lora_rank=None` so a plain `q_proj`, the same
    `kv_a_proj_with_mqa` (512 + 64 out), `kv_a_layernorm`, `kv_b_proj`
    (32 * (128 + 128) out) and `o_proj` -- and sets the same
    `scaling = qk_head_dim ** -0.5`.

    `rope_interleave=False` because `tf.rope` is the rotate-half convention and
    DeepSeek-V3 defaults to the interleaved one. This is not a statement about
    Kimi either way: Kimi's MLA is NoPE (`mla_use_nope: true`), so the RoPE'd
    form exercised alongside it is extra coverage of the same score/merge path
    rather than a configuration Kimi ships.
    """
    from transformers import DeepseekV3Config  # noqa: PLC0415

    return DeepseekV3Config(
        vocab_size=32,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_attention_heads,
        kv_lora_rank=config.kv_lora_rank,
        q_lora_rank=None,
        qk_nope_head_dim=config.qk_nope_head_dim,
        qk_rope_head_dim=config.qk_rope_head_dim,
        v_head_dim=config.v_head_dim,
        rms_norm_eps=config.rms_norm_eps,
        rope_interleave=False,
        rope_parameters={"rope_type": "default", "rope_theta": config.rope_theta},
        attention_bias=False,
        attn_implementation="eager",
    )


def build_mla_attention(seed=0, device="cpu", dtype=DTYPE, config: KimiLinearConfig = CONFIG):
    """A `DeepseekV3Attention` with random weights at a fixed seed."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: PLC0415
        DeepseekV3Attention,
    )

    cfg = build_mla_hf_config(config)
    return oracle.randomised(
        lambda: DeepseekV3Attention(cfg, layer_idx=0), seed, device, dtype
    )


def rope_caches(total: int, config: KimiLinearConfig = CONFIG, device="cpu", dtype=DTYPE):
    """cos / sin caches `[total, qk_rope_head_dim]` for the RoPE'd MLA form.

    Built directly rather than through `DeepseekV3RotaryEmbedding`, because that
    class sizes its inverse frequencies from `config.head_dim` while MLA rotates
    only `qk_rope_head_dim` of each head.
    """

    half = config.qk_rope_head_dim // 2
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    positions = torch.arange(total, dtype=torch.float32, device=device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return (cos, sin) if dtype is None else (cos.to(dtype), sin.to(dtype))


def identity_rope_caches(
    total: int, config: KimiLinearConfig = CONFIG, device="cpu", dtype=DTYPE
):
    """cos = 1, sin = 0: the rotary that leaves q and k untouched.

    This is how `mla_use_nope: true` is expressed without a second attention
    implementation. `apply_rotary_pos_emb(x, x, ones, zeros)` returns
    `x * 1 + rotate_half(x) * 0`, so it is *exactly* the identity by arithmetic.
    That was once measured here (max abs diff 0.0 on both q and k); the
    measurement compared Hugging Face against itself, asking nothing of this
    repository's kernels, and did not survive the move to the shipped source.

    vLLM's `KimiMLAAttention` sets `rotary_emb=None` yet keeps
    `qk_head_dim = 192` and `kv_a_proj_with_mqa` at `512 + 64` out, so NoPE
    does not remove the 64 dimensions -- it only stops rotating them, and they
    still enter the score and the scaling denominator.
    """

    shape_2d = (total, config.qk_rope_head_dim)
    cos = torch.ones(shape_2d, dtype=torch.float32, device=device)
    sin = torch.zeros(shape_2d, dtype=torch.float32, device=device)
    return (cos, sin) if dtype is None else (cos.to(dtype), sin.to(dtype))


def rms_norm(hidden, weight, config: KimiLinearConfig = CONFIG):
    """`tf.rms_norm`'s semantics in torch: `x * rsqrt(mean(x**2) + eps) * weight`.

    The HIR fuses the pre-attention (or post-attention) RMSNorm into its kernel,
    so the oracle has to be fed the states that norm produces. Feeding it the raw
    states instead is not a small error: RMSNorm is scale-invariant, so a second
    RMSNorm downstream -- MLA has one, on the latent -- absorbs the difference and
    only the paths that bypass it (MLA's shared rope part, the MoE router) come
    out wrong, by the reciprocal of the input RMS.
    """

    with torch.no_grad():
        x = hidden.float()
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(ms + config.rms_norm_eps) * weight.float()
    return out.to(hidden.dtype)


def mla_key_value(attention, hidden, cos, sin, config: KimiLinearConfig = CONFIG):
    """MLA's own `(key, value)` for *hidden*, head-major, rotary already applied.

    This is the one step no shared helper can do, because MLA's key is not a
    projection: it is the latent, normed, expanded by `kv_b_proj`, split from the
    value, and concatenated with a rope part that is shared across heads
    (`kv_a_proj_with_mqa` produces one 64-wide rope vector per token, which
    `DeepseekV3Attention` then expands over all 32 heads). Mirrors
    `DeepseekV3Attention.forward` lines 430-446.

    *cos* / *sin* are the exact `[seq, qk_rope_head_dim]` caches this context
    covers. The context starts at absolute position 0, so each row is already at
    the position the projection needs.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    batch, seq = hidden.shape[:2]
    with torch.no_grad():
        compressed = attention.kv_a_proj_with_mqa(hidden)
        latent, k_rot = torch.split(
            compressed, [config.kv_lora_rank, config.qk_rope_head_dim], dim=-1
        )
        k_pass = (
            attention.kv_b_proj(attention.kv_a_layernorm(latent))
            .view(batch, seq, -1, config.qk_nope_head_dim + config.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value = torch.split(
            k_pass, [config.qk_nope_head_dim, config.v_head_dim], dim=-1
        )
        k_rot = k_rot.view(batch, 1, seq, config.qk_rope_head_dim)
        _q, k_rot = apply_rotary_pos_emb(k_rot, k_rot, cos.unsqueeze(0), sin.unsqueeze(0))
        k_rot = k_rot.expand(*k_nope.shape[:-1], -1)
        key = torch.cat([k_nope, k_rot], dim=-1)
    return key, value


def mla_context_kv(attention, hidden_ctx, cos, sin, config: KimiLinearConfig = CONFIG):
    """The cache *attention* would hold for *hidden_ctx*, as explicit tensors.

    `[1, ctx_len, n_heads, dim]`, the kernels' layout. No `Cache` object on
    either side -- the tensors are built by running the module's own projections
    over the context, which is what its cache would have contained.
    """
    key, value = mla_key_value(attention, hidden_ctx, cos, sin, config)
    return key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous()


def mla_decode_reference(attention, hidden_ctx, hidden_new, cos, sin):
    """What MLA produces for *hidden_new* decoded after *hidden_ctx*.

    The whole sequence under a causal mask with the last position kept, so the
    reference never constructs a cache. Causality makes that position's output
    depend on exactly the context before it.

    *cos* / *sin* arrive with exactly one row per sequence position.
    """

    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    mask = oracle.causal_mask(total, hidden_ctx.device.type, hidden_ctx.dtype)
    sequence = torch.cat([hidden_ctx, hidden_new], dim=1)
    with torch.no_grad():
        out, _ = attention(
            sequence,
            position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=mask,
        )
    return out[:, hidden_ctx.shape[1] :, :]


# ── the MoE oracle: DeepseekV3TopkRouter at Kimi's numbers ────────────────────


class MoERouterConfig:
    """What `DeepseekV3TopkRouter` reads, at Kimi's numbers.

    `n_group = topk_group = 1` makes DeepSeek-V3's group-limited routing the
    identity -- every expert is in the one group, so the group mask is all ones
    and nothing is masked to `-inf`. That is why the HIR below has no group
    stage: at these numbers there is nothing for it to do, not because it was
    dropped.
    """

    def __init__(self, config: KimiLinearConfig = CONFIG):
        self.num_experts_per_tok = config.num_experts_per_token
        self.num_local_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = True  # moe_renormalize: true


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
        return (torch.randn(*sizes, device=device) * sigma).to(DTYPE)

    window = SHORT_CONV_KERNEL_SIZE - 1
    return KdaStepInputs(
        args=(
            drawn(1, SEQ_LEN, CONFIG.hidden_size),
            torch.ones(CONFIG.hidden_size, device=device, dtype=DTYPE),
            drawn(1, CONFIG.hidden_size, KDA_PROJ),
            drawn(1, CONFIG.hidden_size, KDA_PROJ),
            drawn(1, CONFIG.hidden_size, KDA_PROJ),
            drawn(SHORT_CONV_KERNEL_SIZE, KDA_PROJ),
            drawn(SHORT_CONV_KERNEL_SIZE, KDA_PROJ),
            drawn(SHORT_CONV_KERNEL_SIZE, KDA_PROJ),
            drawn(1, window, KDA_PROJ),
            drawn(1, window, KDA_PROJ),
            drawn(1, window, KDA_PROJ),
            drawn(1, CONFIG.hidden_size, KDA_HEAD_DIM),
            drawn(1, KDA_HEAD_DIM, KDA_PROJ),
            drawn(KDA_PROJ),
            drawn(KDA_NUM_HEADS),
            drawn(1, CONFIG.hidden_size, KDA_NUM_HEADS),
            drawn(1, CONFIG.hidden_size, KDA_HEAD_DIM),
            drawn(1, KDA_HEAD_DIM, KDA_PROJ),
            torch.ones(KDA_HEAD_DIM, device=device, dtype=DTYPE),
            drawn(1, KDA_PROJ, CONFIG.hidden_size),
            drawn(1, KDA_NUM_HEADS, KDA_HEAD_DIM, KDA_HEAD_DIM),
            torch.full((1, 1, 1), KDA_SCALING, device=device, dtype=DTYPE),
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


# ── KDA: no oracle ───────────────────────────────────────────────────────────


class KdaReferenceUnavailable(RuntimeError):
    """Raised instead of returning inputs there is no oracle to score."""


#: Why the KDA reference is blocked, as measured on 2026-07-28.
#:
#: It is the *reference* that is blocked, not the model: `model.py`
#: describes `kda_attention` completely, and it analyses. What is
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
    `cos = 1, sin = 0`, which `identity_rope_caches` above explains is the
    identity by arithmetic rather than by measurement.
    """
    attention = build_mla_attention(seed=WEIGHT_SEED, device=device)
    caches = identity_rope_caches if nope else rope_caches

    torch.manual_seed(ACTIVATION_SEED)
    drawn = (torch.randn(1, ctx_len + 1, CONFIG.hidden_size, device=device) * 0.1).to(DTYPE)
    hidden_ctx, hidden_new = drawn[:, :ctx_len], drawn[:, ctx_len:]

    # The input RMSNorm belongs to the decoder layer, not to DeepseekV3Attention.
    # The HIR fuses it, so the oracle is fed exactly the states that norm
    # produces. `gamma_in` is drawn rather than set to ones for two reasons: ones
    # would leave a bug in the norm's weight application invisible, and -- because
    # RMSNorm is scale-invariant -- a norm the oracle does not also apply is
    # absorbed by MLA's latent norm and shows up only on the shared rope path.
    gamma_in = (torch.randn(CONFIG.hidden_size, device=device) * 0.1 + 1.0).to(DTYPE)
    normed_ctx = rms_norm(hidden_ctx, gamma_in, CONFIG)

    context_cos, context_sin = caches(ctx_len, CONFIG, device=device)
    k_cache, v_cache = mla_context_kv(
        attention, normed_ctx, context_cos, context_sin, CONFIG
    )

    # The caller selects the decoded token's absolute-position row, so the
    # rank-2 cache is indexed from zero inside the kernel.
    cos, sin = caches(ctx_len + 1, CONFIG, device=device)
    cos, sin = cos[-1:], sin[-1:]
    pos_ids = torch.zeros(1, device=device, dtype=torch.int32)
    scale = torch.full((1, 1, 1, 1), MLA_SCALING, device=device, dtype=DTYPE)

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
    caches = identity_rope_caches if inputs.nope else rope_caches
    cos, sin = caches(inputs.ctx_len + 1, CONFIG, device=inputs.hidden_new.device)
    return mla_decode_reference(
        inputs.attention,
        rms_norm(inputs.hidden_ctx, inputs.gamma_in, CONFIG),
        rms_norm(inputs.hidden_new, inputs.gamma_in, CONFIG),
        cos,
        sin,
    )


def mla_appended_cache_oracle(inputs: MlaStepInputs):
    """The cache the step's caller should hold afterwards.

    Built the same way the input cache was, over the context with the decoded
    token appended: the step's returned key and value are correct exactly when
    appending them reproduces this.
    """
    caches = identity_rope_caches if inputs.nope else rope_caches
    cos, sin = caches(inputs.ctx_len + 1, CONFIG, device=inputs.hidden_new.device)
    return mla_context_kv(
        inputs.attention,
        rms_norm(
            torch.cat([inputs.hidden_ctx, inputs.hidden_new], dim=1),
            inputs.gamma_in,
            CONFIG,
        ),
        cos,
        sin,
        CONFIG,
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
):
    """A `DeepseekV3MoE` at Kimi's numbers, with a NONZERO router bias.

    The nonzero `e_score_correction_bias` is load-bearing and must not be
    "simplified" to the zeros buffer the class defaults to. The router selects on
    `sigmoid(logits) + bias` but takes the routing weights from the *unbiased*
    scores, so at bias = 0 an implementation that gathered the biased scores is
    indistinguishable from a correct one. Measured: with the bias drawn nonzero
    the selected expert set changes for 16/16 tokens and gathering the biased
    scores instead moves the weights by 1.08e-01; at bias = 0 it moves them by
    exactly 0. That the bias is load-bearing is held by
    `smoke_kimi_linear_48b_a3b.py::test_the_router_and_the_shared_expert_are_load_bearing`,
    which zeroes it and requires the comparison to break.
    """
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: PLC0415
        DeepseekV3MoE,
    )

    config = config or CONFIG
    cfg = build_mla_hf_config(config)
    cfg.num_local_experts = config.num_experts
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
    moe = moe.to(DTYPE)
    return moe


def moe_inputs(
    *, act_seed: int = ACTIVATION_SEED, device: str = "cuda", seed: int = WEIGHT_SEED,
    hf_moe=None,
) -> MoeInputs:
    """One deterministic MoE call for one token.

    *hf_moe* may be passed in to reuse an already-built module: at 256 experts its
    weights are about 7 GB, so rebuilding it per draw dominates the test.
    """
    moe = (
        build_hf_moe(seed=seed, device=device, config=CONFIG)
        if hf_moe is None
        else hf_moe
    )

    # The post-attention RMSNorm belongs to the layer; the HIR fuses it, so the
    # oracle is fed exactly the states that norm produces. Drawn rather than ones:
    # the router reads the normed states directly, with no scale-invariant stage
    # downstream to absorb a mismatch.
    torch.manual_seed(seed + 7919)
    gamma_post = (torch.randn(CONFIG.hidden_size, device=device) * 0.1 + 1.0).to(DTYPE)

    torch.manual_seed(act_seed)
    hidden = (torch.randn(1, SEQ_LEN, CONFIG.hidden_size, device=device) * 0.1).to(DTYPE)
    normed = rms_norm(hidden, gamma_post, CONFIG)

    gate_up = moe.experts.gate_up_proj
    w_gate = gate_up[:, : CONFIG.moe_intermediate_size, :].contiguous()
    w_up = gate_up[:, CONFIG.moe_intermediate_size :, :].contiguous()
    w_down = moe.experts.down_proj.contiguous()

    shared = moe.shared_experts
    return MoeInputs(
        args=(
            hidden,
            gamma_post,
            moe.gate.weight.t().contiguous(),
            moe.gate.e_score_correction_bias,
            torch.full((1, 1), CONFIG.routed_scaling_factor, device=device, dtype=DTYPE),
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
