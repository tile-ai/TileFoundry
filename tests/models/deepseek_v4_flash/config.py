"""DeepSeek-V4-Flash dimensions, sourced from ``transformers.AutoConfig`` over
the model's own checked-in ``config.json``; ``REAL`` is the real checkpoint's
shape, ``tiny()`` a small shape for end-to-end tests."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig

#: The file this package's dimensions are read from, and its digest.
#:
#: The provenance here is the artifact rather than a URL: the configuration is
#: checked in beside this module, so what has to be pinned is the file itself, and a
#: digest of a file in the repository is verifiable by anyone with the repository.
#: No public repository is named because none was read -- naming one would suggest a
#: fetch that never happened.
SOURCE_FILE = "config.json"
#: sha256 of that file as checked in.
SOURCE_SHA256 = "52b5a1aa87606cb5be4f3158d706594edb1c4ce97ce6b1cd6079f15df075d7f5"

# Sourced from the modeling code, not config.json: the KV-cache fp8 fake-quant
# block and the e4m3 grid it quantizes onto (hf_attention_ref._fake_quant_fp8_block).
KV_QUANT_BLOCK = 64
FP8E4M3_MAX = 448.0
FP8E4M3_QUANT_EPS = 1e-4  # amax floor, guards log2(0) on an all-zero block


@dataclass(frozen=True)
class DSV4Config:
    """One decoder layer's shape, plus the model-wide embedding/head shape."""

    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window: int
    vocab: int
    moe_inter: int
    n_routed: int
    n_act: int
    route_scale: float
    swiglu_limit: float
    n_layers: int
    n_hash_layers: int
    rms_eps: float
    quant_block: int
    compress_ratios: tuple[int, ...]

    @classmethod
    def from_hf_config(cls, cfg) -> "DSV4Config":
        block = cfg.quantization_config["weight_block_size"]
        if block[0] != block[1]:
            raise ValueError(f"non-square weight_block_size {block} is not supported")
        return cls(
            dim=cfg.hidden_size,
            n_heads=cfg.num_attention_heads,
            n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rope_dim=cfg.qk_rope_head_dim,
            q_lora_rank=cfg.q_lora_rank,
            o_groups=cfg.o_groups,
            o_lora_rank=cfg.o_lora_rank,
            window=cfg.sliding_window,
            vocab=cfg.vocab_size,
            moe_inter=cfg.moe_intermediate_size,
            n_routed=cfg.n_routed_experts,
            n_act=cfg.num_experts_per_tok,
            route_scale=cfg.routed_scaling_factor,
            swiglu_limit=cfg.swiglu_limit,
            n_layers=cfg.num_hidden_layers,
            n_hash_layers=cfg.mlp_layer_types.count("hash_moe"),
            rms_eps=cfg.rms_norm_eps,
            quant_block=block[0],
            compress_ratios=tuple(cfg.compress_rates.get(t, 0) for t in cfg.layer_types),
        )

    @classmethod
    def tiny(cls) -> "DSV4Config":
        """The same model, small enough to run end to end; every dimension
        stays divisible by ``quant_block`` / ``KV_QUANT_BLOCK`` as the real
        shape rules require."""
        return cls(
            dim=256,
            n_heads=2,
            n_kv_heads=1,
            head_dim=256,
            rope_dim=64,
            q_lora_rank=128,
            o_groups=2,
            o_lora_rank=128,
            window=8,
            vocab=64,
            moe_inter=256,
            n_routed=4,
            n_act=2,
            route_scale=1.5,
            swiglu_limit=10.0,
            n_layers=1,
            n_hash_layers=1,
            rms_eps=1e-6,
            quant_block=128,
            compress_ratios=(0,),
        )

    # ── derived shapes ───────────────────────────────────────────────────────

    @property
    def rope_half(self) -> int:
        return self.rope_dim // 2

    @property
    def nope_dim(self) -> int:
        """Head dims left unrotated, and the only part the KV cache quantizes."""
        return self.head_dim - self.rope_dim

    @property
    def kv_quant_blocks(self) -> int:
        return self.nope_dim // KV_QUANT_BLOCK

    @property
    def max_ctx(self) -> int:
        """The longest context a decode step can be asked about.

        The window, not the position embedding: a query in a sliding layer
        attends ``window`` positions counting its own, so the context before it
        is one shorter. A longer cache is a context this layer type does not
        attend rather than one it attends slowly.
        """
        return self.window - 1

    @property
    def q_proj(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def wo_a_in(self) -> int:
        return self.q_proj // self.o_groups

    @property
    def wo_a_out(self) -> int:
        return self.o_groups * self.o_lora_rank

    def blocks(self, extent: int) -> int:
        """Block-scale count along an axis of *extent* (weight_block_size)."""
        if extent % self.quant_block:
            raise ValueError(
                f"extent {extent} is not a multiple of quant_block {self.quant_block}"
            )
        return extent // self.quant_block


HF_CONFIG = AutoConfig.from_pretrained(Path(__file__).parent)
REAL = DSV4Config.from_hf_config(HF_CONFIG)

#: The same model at a shape small enough to build and run end to end. Stated
#: here beside `REAL`, so the two shapes this model is asked about are read from
#: one place rather than constructed at each call site.
TINY = DSV4Config.tiny()

#: The layer the attention description is authored at: the first sliding-window
#: one, which is the layer type that carries no compressor.
SLIDING_LAYER = HF_CONFIG.layer_types.index("sliding_attention")


def fake_quant_kv(latent):
    """*latent*'s unrotated head dims, through the checkpoint's fp8 KV round trip.

    The one place a reference for this model is not Hugging Face's own module.
    V4-Flash stores its KV latent as fp8 e4m3 with a per-block power-of-two
    (ue8m0) scale, which the official inference path does and
    ``modeling_deepseek_v4`` does not; a reference without it would be scoring
    the kernel against a model that keeps precision the kernel is specified to
    throw away. Only the unrotated dims are quantized -- the rope slice stays
    bf16, as it does in the kernel.
    """
    import torch  # noqa: PLC0415

    nope, rope = latent[..., : REAL.nope_dim], latent[..., REAL.nope_dim :]
    blocks = nope.float().reshape(*nope.shape[:-1], REAL.kv_quant_blocks, KV_QUANT_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(FP8E4M3_QUANT_EPS)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8E4M3_MAX)))
    scaled = (blocks / scale).clamp(-FP8E4M3_MAX, FP8E4M3_MAX)
    dequant = scaled.to(torch.float8_e4m3fn).to(torch.float32) * scale
    return torch.cat([dequant.reshape(nope.shape).to(latent.dtype), rope], dim=-1)


def build_hf_attention(seed=0, device="cuda", dtype=None):
    """A ``DeepseekV4Attention`` for the sliding layer, random at a fixed seed.

    The fp8 KV round trip is installed on the module's own KV norm rather than
    applied by whoever calls it, so every path that reads this layer's stored
    latent -- the cache built from a context, and the full-sequence forward the
    oracle takes its answer from -- stores the same thing the kernel does.
    """
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
        DeepseekV4Attention,
    )

    from tests.models import decode_oracle as oracle  # noqa: PLC0415

    layer = oracle.randomised(
        lambda: DeepseekV4Attention(HF_CONFIG, layer_idx=SLIDING_LAYER),
        seed, device, dtype,
    )
    layer.kv_norm.register_forward_hook(lambda _m, _args, out: fake_quant_kv(out))
    return layer


def rope_caches(total: int, device="cuda"):
    """Interleaved cos / sin ``[1, total, rope_half]`` for the ``main`` rope label.

    One entry per rotated pair, which is what V4's interleaved rotation takes --
    and why ``decode_oracle.rope_caches`` cannot build it: this model's rotary
    embedding is keyed by layer type and returns the half-width pair, not the
    duplicated full-width one every other model in the corpus uses.
    """
    import torch  # noqa: PLC0415
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
        DeepseekV4RotaryEmbedding,
    )

    rotary = DeepseekV4RotaryEmbedding(HF_CONFIG).to(device)
    reference = torch.zeros(1, total, REAL.dim, device=device)
    positions = torch.arange(total, device=device).unsqueeze(0)
    return rotary(reference, positions, layer_type="main")


def context_kv(layer, hidden_ctx):
    """The KV cache *layer* would hold for *hidden_ctx*, as an explicit tensor.

    Its own norm, its own projection, its own rotation, in its own order -- the
    same tensors its cache would have held, in the kernels'
    ``[1, ctx_len, n_kv_heads, head_dim]`` layout rather than Hugging Face's
    head-major one. ``decode_oracle.context_kv`` states this for a model with a
    separate key and value drawn through a pair-wise rotary; V4 has one shared
    latent and a single-tensor rotation, so the shape of that hook does not fit.
    """
    import torch  # noqa: PLC0415
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    ctx_len = hidden_ctx.shape[1]
    cos, sin = rope_caches(ctx_len, hidden_ctx.device.type)
    with torch.no_grad():
        latent = layer.kv_norm(layer.kv_proj(hidden_ctx))
        latent = latent.view(1, ctx_len, 1, REAL.head_dim).transpose(1, 2)
        rotated = apply_rotary_pos_emb(latent, cos.to(latent.dtype), sin.to(latent.dtype))
    return rotated.transpose(1, 2).contiguous()


def decode_reference(layer, hidden_ctx, hidden_new):
    """What *layer* produces for *hidden_new* decoded after *hidden_ctx*.

    The whole sequence under a causal mask, last position kept, no ``Cache``
    object on either side. Causal is the whole story only while the sequence
    fits the window, which is why the context is asked to.
    """
    import torch  # noqa: PLC0415

    from tests.models import decode_oracle as oracle  # noqa: PLC0415

    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    if total > REAL.window:
        raise ValueError(
            f"a sliding layer attends {REAL.window} positions counting its own; "
            f"a {total}-long sequence is not a decode step it can take"
        )
    device = hidden_ctx.device.type
    cos, sin = rope_caches(total, device)
    mask = oracle.causal_mask(total, device, hidden_ctx.dtype)
    positions = torch.arange(total, device=device).unsqueeze(0)
    with torch.no_grad():
        out, _ = layer(
            torch.cat([hidden_ctx, hidden_new], dim=1),
            position_embeddings={"main": (cos.to(hidden_ctx.dtype), sin.to(hidden_ctx.dtype))},
            position_ids=positions,
            attention_mask=mask,
        )
    return out[:, hidden_ctx.shape[1] :, :]


__all__ = [
    "FP8E4M3_MAX",
    "FP8E4M3_QUANT_EPS",
    "HF_CONFIG",
    "KV_QUANT_BLOCK",
    "REAL",
    "SLIDING_LAYER",
    "TINY",
    "DSV4Config",
    "build_hf_attention",
    "context_kv",
    "decode_reference",
    "fake_quant_kv",
    "rope_caches",
]
