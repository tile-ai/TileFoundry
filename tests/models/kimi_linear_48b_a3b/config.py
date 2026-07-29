"""Kimi-Linear-48B-A3B dimensions, its provenance, and the oracles it is held to.

Provenance is recorded because this model's own modelling code is NOT available
here. `transformers` 5.14.1 has no `kimi_linear`: `KimiLinearForCausalLM` appears
nowhere in the installed package, `kimi_linear` is absent from `CONFIG_MAPPING`,
and the published config's `auto_map` points at `configuration_kimi.py` /
`modeling_kimi.py`, which are remote code. Measured, offline:

    AutoConfig.from_pretrained(<dir>, trust_remote_code=False)
      -> ValueError: ... contains custom code which must be executed
    AutoConfig.from_pretrained(<dir>, trust_remote_code=True)
      -> OSError: does not appear to have a file named configuration_kimi.py

So transformers cannot build even the *config*, let alone the model. Every number
below therefore comes from the published `config.json`, pinned by revision and
digest, and every oracle below is a *different* model's Hugging Face module that
has been shown to compute the same function -- never a re-derivation of Kimi's
semantics from these numbers, which would compare two guesses.

Four things the config alone actively contradicts, so they are recorded here
rather than left to be rediscovered (all read off vLLM 0.18.0's
`model_executor/models/kimi_linear.py`, which is on this machine but not
importable -- an orphaned python3.13 site-packages under a 3.12 interpreter):

- `kda_layers` / `full_attn_layers` are **1-BASED**. `is_kda_layer` returns
  `(layer_idx + 1) in kda_layers`, so 0-based layer 0 is KDA and 0-based layer 3
  is the first full-attention (MLA) layer.
- With `first_k_dense_replace: 1`, 0-based layer 0 is KDA + a **dense** MLP while
  layer 3 is MLA + MoE. A two-layer minimum therefore cannot be layers 0 and 1:
  layer 1 adds MoE but no new attention kind.
- Top-level `head_dim: 72` is read by **neither** path. It is just
  `hidden_size // num_attention_heads`; KDA uses `linear_attn_config.head_dim`
  = 128 and MLA uses 192 (q/k) and 128 (v).
- `moe_renormalize: true` means normalise-then-scale: the top-k weights are
  divided by their sum and only then multiplied by `routed_scaling_factor`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from tests.models import decode_oracle as oracle

# ── provenance ───────────────────────────────────────────────────────────────

#: Where the numbers below come from.
SOURCE_URL = "https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json"

#: The exact commit those numbers were read at.
SOURCE_REVISION = "e1df551a447157d4658b573f9a695d57658590e9"

#: sha256 of the whole published config.json at that revision.
SOURCE_SHA256 = "a6ac3c2c4b5aa72370f9727f49ffa4432715d20061889acdb37c688be853096e"

#: Exactly the published fields this package depends on, copied verbatim. A
#: digest over this -- rather than over the whole file -- is what the package
#: actually rests on: an upstream edit to a field nobody here reads should not
#: invalidate the model, and an edit to one of these must.
DEPENDED_FIELDS: dict[str, object] = {
    "architectures": ["KimiLinearForCausalLM"],
    "model_type": "kimi_linear",
    "hidden_size": 2304,
    "intermediate_size": 9216,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "num_hidden_layers": 27,
    "rms_norm_eps": 1e-05,
    "rope_theta": 10000.0,
    "vocab_size": 163840,
    "hidden_act": "silu",
    # MLA
    "kv_lora_rank": 512,
    "q_lora_rank": None,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": True,
    # KDA
    "linear_attn_config": {
        "head_dim": 128,
        "num_heads": 32,
        "short_conv_kernel_size": 4,
        "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,
                       17, 18, 19, 21, 22, 23, 25, 26],
        "full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
    },
    # MoE
    "num_experts": 256,
    "num_experts_per_token": 8,
    "num_shared_experts": 1,
    "moe_intermediate_size": 1024,
    "moe_router_activation_func": "sigmoid",
    "moe_renormalize": True,
    "routed_scaling_factor": 2.446,
    "use_grouped_topk": True,
    "num_expert_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "num_nextn_predict_layers": 0,
}


def depended_digest() -> str:
    """sha256 over `DEPENDED_FIELDS`, canonically encoded."""
    encoded = json.dumps(DEPENDED_FIELDS, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


#: The digest of the fields above as this package was written against them.
DEPENDED_SHA256 = "9b4bf4e5d53bbc284dc50fbd0f5951ad4d9835d87d99a59f62556bfdbdbeb0de"


# ── the layer taxonomy, 1-based as published ──────────────────────────────────


def is_kda_layer(layer_idx: int) -> bool:
    """Whether 0-based *layer_idx* is a KDA layer.

    The `+ 1` is not an off-by-one: the published lists are 1-based, and vLLM's
    `KimiLinearConfig.is_kda_layer` reads them the same way.
    """
    return (layer_idx + 1) in DEPENDED_FIELDS["linear_attn_config"]["kda_layers"]


def is_dense_layer(layer_idx: int) -> bool:
    """Whether 0-based *layer_idx* uses a dense MLP rather than the MoE."""
    return layer_idx < DEPENDED_FIELDS["first_k_dense_replace"]


#: The two 0-based layer indices a minimum model must contain: one KDA layer and
#: one MLA layer. Layer 0 is KDA + dense MLP; layer 3 is MLA + MoE. Layers 1 and
#: 2 are KDA + MoE and add no attention kind, so they are not replicated.
MINIMUM_LAYERS = (0, 3)


# ── dimensions ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KimiLinearConfig:
    """The published fields this model is built at, plus the envelope we choose.

    A typed subset of the published `config.json`, under the published names, and
    not an implementation of `transformers.PretrainedConfig`: this model's own
    config class is remote code that is not available here (see this module's
    docstring), so anything claiming to be it would be a counterfeit. What this
    holds is exactly the fields the HIR and the oracles read.

    No field carries a default. Every published number has to arrive from the
    published fields, so a value edited upstream cannot be silently replaced by a
    stale one written here -- see `from_published`. The two envelope fields and
    the dtype are ours rather than published, so they are stated at the call.

    Anything computable from these is a property, not a field: a derived number
    stored twice can disagree with itself.
    """

    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float

    # MLA. `qk_head_dim` is the score dimension and therefore the scaling
    # denominator: 128 + 64 = 192, NOT v_head_dim. See `mla_scaling`.
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    # KDA. Published nested under `linear_attn_config`, so the prefix is ours: a
    # flat `head_dim` would collide with MLA's, which is a different number.
    kda_head_dim: int
    kda_num_heads: int
    short_conv_kernel_size: int

    # MoE
    num_experts: int
    num_experts_per_token: int
    num_shared_experts: int
    moe_intermediate_size: int
    routed_scaling_factor: float

    #: Envelope for the one dynamic dimension, and the position table's extent.
    #: Ours, not published: the published `max_position_embeddings` is 1048576,
    #: and the envelope only has to contain the lengths anything here is asked
    #: about -- a million-long bound costs analysis precision for nothing.
    max_ctx: int
    max_pos: int

    #: The dtype the HIR is authored in. Also ours: the published checkpoint's
    #: storage dtype is a separate question from what these kernels compute in.
    dtype: str

    @classmethod
    def from_published(
        cls,
        published: dict[str, object],
        *,
        max_ctx: int,
        max_pos: int,
        dtype: str,
    ) -> "KimiLinearConfig":
        """Read the published fields, and take the envelope from the caller.

        Reading rather than restating is the point: `published` is the digested
        subset this package is pinned to, so a field that moves upstream moves
        here, and one that is missing raises instead of falling back.
        """
        linear = published["linear_attn_config"]
        return cls(
            hidden_size=published["hidden_size"],
            num_attention_heads=published["num_attention_heads"],
            intermediate_size=published["intermediate_size"],
            rms_norm_eps=published["rms_norm_eps"],
            rope_theta=published["rope_theta"],
            kv_lora_rank=published["kv_lora_rank"],
            qk_nope_head_dim=published["qk_nope_head_dim"],
            qk_rope_head_dim=published["qk_rope_head_dim"],
            v_head_dim=published["v_head_dim"],
            kda_head_dim=linear["head_dim"],
            kda_num_heads=linear["num_heads"],
            short_conv_kernel_size=linear["short_conv_kernel_size"],
            num_experts=published["num_experts"],
            num_experts_per_token=published["num_experts_per_token"],
            num_shared_experts=published["num_shared_experts"],
            moe_intermediate_size=published["moe_intermediate_size"],
            routed_scaling_factor=published["routed_scaling_factor"],
            max_ctx=max_ctx,
            max_pos=max_pos,
            dtype=dtype,
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def q_proj(self) -> int:
        return self.num_attention_heads * self.qk_head_dim

    @property
    def kv_a_proj(self) -> int:
        """`kv_a_proj_with_mqa` output: the latent plus the shared rope part."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kv_b_proj(self) -> int:
        return self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim)

    @property
    def v_proj(self) -> int:
        return self.num_attention_heads * self.v_head_dim

    @property
    def kda_proj(self) -> int:
        return self.kda_num_heads * self.kda_head_dim

    @property
    def shared_intermediate(self) -> int:
        return self.moe_intermediate_size * self.num_shared_experts

    @property
    def mla_scaling(self) -> float:
        """`qk_head_dim ** -0.5`.

        Measured, not assumed: 192 ** -0.5 = 0.0721688. The natural guess from
        the config alone -- `v_head_dim ** -0.5` = 0.0883883 -- is 22.5% off, and
        nothing in the config says which is meant. vLLM's `KimiMLAAttention` and
        `DeepseekV3Attention` both use `qk_head_dim ** -0.5`.
        """
        return self.qk_head_dim ** -0.5

    @property
    def kda_scaling(self) -> float:
        """`kda_head_dim ** -0.5`, applied to q *after* its l2 normalisation."""
        return self.kda_head_dim ** -0.5


#: The published model. The envelope is stated here rather than defaulted, so the
#: numbers that are ours and not the checkpoint's are visible where they are chosen.
REAL = KimiLinearConfig.from_published(
    DEPENDED_FIELDS, max_ctx=4096, max_pos=4096, dtype="f32"
)

#: The published config with a quarter of the experts. The MoE oracle needs one
#: expert's weights per expert on the device, and at 256 that is about 7 GB in
#: f32 while the suite runs eight ways in parallel; the comparison it makes --
#: that the router picks the same experts and weights them the same way -- is the
#: same comparison at 64. Nothing else moves, so a difference between the two is
#: the expert count and not a second config.
SMALL_MOE = replace(REAL, num_experts=64)

#: One token per decode step.
SEQ_LEN = 1


# ── the MLA oracle: DeepseekV3Attention at Kimi's ranks ───────────────────────


def build_mla_hf_config(config: KimiLinearConfig = REAL):
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


def build_mla_attention(seed=0, device="cpu", dtype=None, config: KimiLinearConfig = REAL):
    """A `DeepseekV3Attention` with random weights at a fixed seed."""
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: PLC0415
        DeepseekV3Attention,
    )

    cfg = build_mla_hf_config(config)
    return oracle.randomised(
        lambda: DeepseekV3Attention(cfg, layer_idx=0), seed, device, dtype
    )


def rope_caches(config: KimiLinearConfig = REAL, device="cpu", dtype=None):
    """cos / sin caches `[max_pos, qk_rope_head_dim]` for the RoPE'd MLA form.

    Built directly rather than through `DeepseekV3RotaryEmbedding`, because that
    class sizes its inverse frequencies from `config.head_dim` while MLA rotates
    only `qk_rope_head_dim` of each head.
    """
    import torch  # noqa: PLC0415

    half = config.qk_rope_head_dim // 2
    inv_freq = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    positions = torch.arange(config.max_pos, dtype=torch.float32, device=device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return (cos, sin) if dtype is None else (cos.to(dtype), sin.to(dtype))


def identity_rope_caches(config: KimiLinearConfig = REAL, device="cpu", dtype=None):
    """cos = 1, sin = 0: the rotary that leaves q and k untouched.

    This is how `mla_use_nope: true` is expressed without a second attention
    implementation. `apply_rotary_pos_emb(x, x, ones, zeros)` returns
    `x * 1 + rotate_half(x) * 0`, and that it is *exactly* the identity is
    measured in `test_mla.py` (max abs diff 0.0 on both q and k), not assumed.

    vLLM's `KimiMLAAttention` sets `rotary_emb=None` yet keeps
    `qk_head_dim = 192` and `kv_a_proj_with_mqa` at `512 + 64` out, so NoPE
    does not remove the 64 dimensions -- it only stops rotating them, and they
    still enter the score and the scaling denominator.
    """
    import torch  # noqa: PLC0415

    shape_2d = (config.max_pos, config.qk_rope_head_dim)
    cos = torch.ones(shape_2d, dtype=torch.float32, device=device)
    sin = torch.zeros(shape_2d, dtype=torch.float32, device=device)
    return (cos, sin) if dtype is None else (cos.to(dtype), sin.to(dtype))


def rms_norm(hidden, weight, config: KimiLinearConfig = REAL):
    """`tf.rms_norm`'s semantics in torch: `x * rsqrt(mean(x**2) + eps) * weight`.

    The HIR fuses the pre-attention (or post-attention) RMSNorm into its kernel,
    so the oracle has to be fed the states that norm produces. Feeding it the raw
    states instead is not a small error: RMSNorm is scale-invariant, so a second
    RMSNorm downstream -- MLA has one, on the latent -- absorbs the difference and
    only the paths that bypass it (MLA's shared rope part, the MoE router) come
    out wrong, by the reciprocal of the input RMS.
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        x = hidden.float()
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(ms + config.rms_norm_eps) * weight.float()
    return out.to(hidden.dtype)


def mla_key_value(attention, hidden, cos, sin, config: KimiLinearConfig = REAL):
    """MLA's own `(key, value)` for *hidden*, head-major, rotary already applied.

    This is the one step no shared helper can do, because MLA's key is not a
    projection: it is the latent, normed, expanded by `kv_b_proj`, split from the
    value, and concatenated with a rope part that is shared across heads
    (`kv_a_proj_with_mqa` produces one 64-wide rope vector per token, which
    `DeepseekV3Attention` then expands over all 32 heads). Mirrors
    `DeepseekV3Attention.forward` lines 430-446.

    *cos* / *sin* are the full `[max_pos, qk_rope_head_dim]` caches the kernel
    takes, sliced here to the positions this call covers. The context starts at
    absolute position 0, so a prefix is the right slice; the kernel makes the same
    selection by `pos_ids` instead.
    """
    import torch  # noqa: PLC0415
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    batch, seq = hidden.shape[:2]
    cos, sin = cos[:seq], sin[:seq]
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


def mla_context_kv(attention, hidden_ctx, cos, sin, config: KimiLinearConfig = REAL):
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

    *cos* / *sin* arrive as the full caches and are sliced to the sequence, as in
    `mla_key_value`.
    """
    import torch  # noqa: PLC0415

    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    cos, sin = cos[:total], sin[:total]
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

    def __init__(self, config: KimiLinearConfig = REAL):
        self.num_experts_per_tok = config.num_experts_per_token
        self.num_local_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = True  # moe_renormalize: true


def build_moe_router(seed=0, device="cpu", config: KimiLinearConfig = REAL):
    """A `DeepseekV3TopkRouter` with random weights AND a nonzero bias.

    The nonzero `e_score_correction_bias` is the point, not a detail. The router
    selects on `scores + bias` but gathers the routing weights from the *unbiased*
    sigmoid scores. At bias = 0 those two are the same tensor, so an
    implementation that gathered the biased scores would pass -- measured: with
    the bias drawn nonzero the selected expert set changes for 16/16 tokens, and
    gathering biased instead of unbiased scores moves the weights by 1.08e-01,
    while at bias = 0 it moves them by exactly 0. Do not "simplify" this to zero.

    vLLM's `KimiMoE` allocates `e_score_correction_bias` as a loaded checkpoint
    parameter, so Kimi genuinely has one; it is not a DeepSeek-only artefact.
    """
    import torch  # noqa: PLC0415
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (  # noqa: PLC0415
        DeepseekV3TopkRouter,
    )

    torch.manual_seed(seed)
    with torch.device(device):
        router = DeepseekV3TopkRouter(MoERouterConfig(config))
    router = router.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        router.weight.normal_(0.0, 0.05)
        router.e_score_correction_bias.normal_(0.0, 0.5)
    return router


def linear_weight(linear):
    """HF `nn.Linear.weight` `[out, in]` -> the kernels' `[1, in, out]`."""
    return oracle.linear_weight(linear)


__all__ = [
    "DEPENDED_FIELDS",
    "DEPENDED_SHA256",
    "MINIMUM_LAYERS",
    "REAL",
    "SMALL_MOE",
    "SEQ_LEN",
    "SOURCE_REVISION",
    "SOURCE_SHA256",
    "SOURCE_URL",
    "KimiLinearConfig",
    "MoERouterConfig",
    "build_mla_attention",
    "build_mla_hf_config",
    "build_moe_router",
    "depended_digest",
    "identity_rope_caches",
    "is_dense_layer",
    "is_kda_layer",
    "linear_weight",
    "mla_context_kv",
    "mla_decode_reference",
    "mla_key_value",
    "rope_caches",
]
