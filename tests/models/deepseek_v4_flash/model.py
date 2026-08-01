"""DeepSeek-V4-Flash as IR Modules: the sliding-window MLA attention submodule,
the two MoE blocks, the decoder layer that composes them, and the causal-LM root.

One authored source at two configurations: the corpus analyses and schedules the
attention submodule at the real checkpoint's dimensions, and
`test_causal_lm_e2e.py` runs the whole tree at a shape small enough to be
affordable. The class bodies sit inside functions rather than at file scope, so
each is evaluated once per configuration, and ``build_deepseek_v4_flash`` is how a
caller names the shape it means. Trees from two calls share no IR node.

Decode, one token per step. The step's own token count is the literal 1, so the
only dimension carried as a range is the context the step reads: ``ctx_len``, the
length of the KV cache handed in. This model's attention is sliding, so that
range is bounded by the window rather than by the position embedding -- a query
attends ``window`` positions counting itself, and a longer cache is a context
this layer cannot attend rather than one it attends slowly.

The cache is explicit tensors in and out, and the two directions are not the same
tensor. ``mla_attend`` reads the context *before* this token -- ``ctx_len``
positions, read-only -- and ``mla_kv_update`` produces this token's own KV
latent, one position, for the caller to append and to evict from at the window
edge. A kernel returning the grown cache would have an axis of ``ctx_len + 1``,
and a sum of a range and a constant cannot feed the matmul that consumes it.

That split is why attention here is an online softmax rather than one ``softmax``
over the cache: the new token attends itself as well as the cache, the two score
groups live in differently shaped tensors, and the per-head attention sink is a
third group of one denominator-only column. Each is reduced to its own ``(max,
sum, weighted values)`` partial and the partials are merged by a log-sum-exp
rescale, in f32 -- what a real decode kernel accumulates in, and what makes a
512-wide dot product mean anything in a bf16 model. No mask is needed: a single
query at the end of the context may attend every position it was given, which is
what makes the window the caller's eviction policy instead of a row of masked-out
slots the kernel scores anyway.

MQA: one shared ``head_dim``-wide KV latent (``n_kv_heads == 1``) read as both
key and value, so the value carries the key's rotation and the output's rope
slice is un-rotated at the query's own position afterwards.

RoPE uses DeepSeek's interleaved-pairs convention (view-as-complex on adjacent
dims), not ``tilefoundry.dsl.tf.rope``'s rotate-half, so it is built from
subscripts and ``tf.reshape``/``tf.concat`` instead.

Weights are derived automatically from every ``ConstTensor`` param. Routed expert
weights are fp8 e4m3 with a ``quant_block``-square ``ue8m0`` (``f8e8m0``) block
scale; only the *scale* tensors need a converter (cast, F32 on disk ->
``f8e8m0``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import AutoConfig

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, ReduceKind, Tensor, tf
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

_MAIN_ROPE: "tuple | None" = None


def _main_rope() -> tuple:
    """HF's own inverse frequencies and attention scaling for the ``main`` rope
    label, computed by ``DeepseekV4RotaryEmbedding`` rather than reproduced.

    A ``sliding_attention`` layer takes ``main`` (plain rope); only the
    compressed layer types take the yarn-scaled ``compress`` label.
    """
    global _MAIN_ROPE
    if _MAIN_ROPE is None:
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (  # noqa: PLC0415
            DeepseekV4RotaryEmbedding,
        )

        _MAIN_ROPE = DeepseekV4RotaryEmbedding.compute_default_rope_parameters(
            HF_CONFIG, layer_type="main"
        )
    return _MAIN_ROPE


def _rope_cos_sin(position: int, *, device):
    """cos / sin for one absolute sequence *position*, each
    ``(config.rope_half,)`` f32, one value per rotated pair."""
    import torch  # noqa: PLC0415

    inv_freq, attention_scaling = _main_rope()
    angles = position * inv_freq.to(torch.float64)
    cos = (angles.cos() * attention_scaling).to(dtype=torch.float32, device=device)
    sin = (angles.sin() * attention_scaling).to(dtype=torch.float32, device=device)
    return cos, sin

#: The context length the decode loop starts from. Producing a context from a
#: prompt is a prefill, which this package does not state, so the loop starts
#: from one drawn at a fixed seed instead.
SEED_CTX_LEN = 1
SEED_CTX_SEED = 20260728



# ── the checkpoint's own configuration, and the shapes it implies ────────────


def published(path: Path | None = None, **overrides):
    """The checkpoint's own configuration, read by the class Hugging Face uses."""
    directory = Path(__file__).parent if path is None else path
    return AutoConfig.from_pretrained(directory, **overrides)


#: Fake-quantisation constants for the KV cache, sourced from the modelling code
#: rather than from `config.json`: the fp8 block and the e4m3 grid it lands on.
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


#: Every dimension below is derived from the published file, and every function
#: that builds IR derives its own from the config it is handed. Nothing is read
#: off this module-level value inside a builder: a shape computed here and used
#: there would be the published shape wearing a caller's config, which is
#: exactly how a small-config tree ends up holding a published-config kernel.
HF_CONFIG = published()
REAL = DSV4Config.from_hf_config(HF_CONFIG)

def _submodules(config: DSV4Config):
    """This model's leaves at *config*: the attention submodule and the two MoE
    blocks, each built fresh for this call."""

    # The prior cache this step reads: the only range this model carries. Zero is a
    # first step, and the exclusive upper bound is the window itself -- a query
    # attends `window` positions counting its own, so the cache before it is one
    # shorter.
    C = DimVar("ctx_len", 0, config.window)


    @module(entry="mla_attend")
    class DeepseekV4Attention:
        @func
        def mla_kv_update(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            gamma_kv: ConstTensor[(config.head_dim,), "bf16"],
            w_kv: ConstTensor[(config.dim, config.head_dim), "bf16"],
            cos_pos: Tensor[(1, 1, 1, config.rope_half), "f32"],
            sin_pos: Tensor[(1, 1, 1, config.rope_half), "f32"],
        ) -> Tensor[(1, 1, 1, config.head_dim), "bf16"]:
            # This token's own KV latent, one position: what the caller appends to
            # the cache it passed `mla_attend`.
            kv = tf.matmul(hidden, w_kv)
            kv_n = tf.rms_norm(kv, gamma_kv)
            kv_4d = tf.reshape(kv_n, new_shape=(1, 1, 1, config.head_dim))
            kv_nope = kv_4d[:, :, :, : config.nope_dim]
            kv_rope_in = kv_4d[:, :, :, config.nope_dim : config.head_dim]

            # FP8 e4m3 fake-quant of the non-rope KV latent: block-absmax, scale
            # rounded up to a power of two (ue8m0), then a real fp8e4m3 round
            # trip. kv_rope_in stays bf16/unquantized.
            kv_nope_f32 = tf.cast(kv_nope, dtype="f32")
            kv_nope_blk = tf.reshape(
                kv_nope_f32, new_shape=(1, 1, 1, config.kv_quant_blocks, KV_QUANT_BLOCK),
            )
            kv_amax = tf.reduce(kv_nope_blk, axes=(-1,), keepdim=True, kind=ReduceKind.ABS_MAX)
            kv_amax = tf.max(kv_amax, FP8E4M3_QUANT_EPS)
            kv_scale = tf.exp2(tf.ceil(tf.log2(kv_amax / FP8E4M3_MAX)))
            kv_scaled = kv_nope_blk / kv_scale
            kv_scaled = tf.min(tf.max(kv_scaled, -FP8E4M3_MAX), FP8E4M3_MAX)
            kv_q_fp8 = tf.cast(kv_scaled, dtype="fp8e4m3")
            kv_dq = tf.cast(kv_q_fp8, dtype="f32") * kv_scale
            kv_nope_q = tf.cast(tf.reshape(kv_dq, new_shape=(1, 1, 1, config.nope_dim)), dtype="bf16")

            kv_r0 = kv_rope_in[:, :, :, 0 : config.rope_dim : 2]
            kv_r1 = kv_rope_in[:, :, :, 1 : config.rope_dim : 2]
            kv_r0_f32 = tf.cast(kv_r0, dtype="f32")
            kv_r1_f32 = tf.cast(kv_r1, dtype="f32")
            kv_o0_f32 = kv_r0_f32 * cos_pos - kv_r1_f32 * sin_pos
            kv_o1_f32 = kv_r0_f32 * sin_pos + kv_r1_f32 * cos_pos
            kv_o0 = tf.cast(kv_o0_f32, dtype="bf16")
            kv_o1 = tf.cast(kv_o1_f32, dtype="bf16")
            kv_o0 = tf.reshape(kv_o0, new_shape=(1, 1, 1, config.rope_half, 1))
            kv_o1 = tf.reshape(kv_o1, new_shape=(1, 1, 1, config.rope_half, 1))
            kv_interleaved = tf.concat(kv_o0, kv_o1, axis=-1)
            kv_rope_out = tf.reshape(kv_interleaved, new_shape=(1, 1, 1, config.rope_dim))
            return tf.concat(kv_nope_q, kv_rope_out, axis=-1)

        @mla_kv_update.converter("w_kv")
        def _(
            wkv_weight: ConstTensor[(config.head_dim, config.dim), "fp8e4m3"],
            wkv_scale: ConstTensor[(config.blocks(config.head_dim), config.blocks(config.dim)), "f32"],
        ) -> Tensor[(config.dim, config.head_dim), "bf16"]:
            # Block dequant, then transpose to the (dim, head_dim) orientation mla_kv_update expects.
            blocks = tf.reshape(
                tf.cast(wkv_weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.head_dim), config.quant_block,
                    config.blocks(config.dim), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(wkv_scale, dtype="bf16"),
                new_shape=(config.blocks(config.head_dim), 1, config.blocks(config.dim), 1),
            )
            dequant = tf.reshape(blocks * block_scale, new_shape=(config.head_dim, config.dim))
            return tf.transpose(dequant, perm=(1, 0))

        @func
        def mla_attend(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            gamma_q_lora: ConstTensor[(config.q_lora_rank,), "bf16"],
            w_q_a: ConstTensor[(config.dim, config.q_lora_rank), "bf16"],
            w_q_b: ConstTensor[(config.q_lora_rank, config.q_proj), "bf16"],
            ones_head_dim: Tensor[(config.head_dim,), "bf16"],
            cos_pos: Tensor[(1, 1, 1, config.rope_half), "f32"],
            sin_pos: Tensor[(1, 1, 1, config.rope_half), "f32"],
            kv_cache: Tensor[(1, C, 1, config.head_dim), "bf16"],
            kv_new: Tensor[(1, 1, 1, config.head_dim), "bf16"],
            attn_sink: ConstTensor[(1, 1, config.n_heads, 1), "f32"],
            scale: Tensor[(1, 1, 1, 1), "bf16"],
            w_o_a: ConstTensor[(config.o_groups, config.wo_a_in, config.o_lora_rank), "bf16"],
            w_o_b: ConstTensor[(config.wo_a_out, config.dim), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            # q_rescaled is a per-head unweighted RMS rescale (tf.rms_norm, all-ones weight).
            q_lat = tf.rms_norm(tf.matmul(hidden, w_q_a), gamma_q_lora)
            q_full = tf.matmul(q_lat, w_q_b)
            q = tf.reshape(q_full, new_shape=(1, 1, config.n_heads, config.head_dim))
            q_rescaled = tf.rms_norm(q, ones_head_dim)
            q_nope = q_rescaled[:, :, :, : config.nope_dim]
            q_rope_in = q_rescaled[:, :, :, config.nope_dim : config.head_dim]
            q_r0 = q_rope_in[:, :, :, 0 : config.rope_dim : 2]
            q_r1 = q_rope_in[:, :, :, 1 : config.rope_dim : 2]
            q_r0_f32 = tf.cast(q_r0, dtype="f32")
            q_r1_f32 = tf.cast(q_r1, dtype="f32")
            q_o0_f32 = q_r0_f32 * cos_pos - q_r1_f32 * sin_pos
            q_o1_f32 = q_r0_f32 * sin_pos + q_r1_f32 * cos_pos
            q_o0 = tf.cast(q_o0_f32, dtype="bf16")
            q_o1 = tf.cast(q_o1_f32, dtype="bf16")
            q_o0 = tf.reshape(q_o0, new_shape=(1, 1, config.n_heads, config.rope_half, 1))
            q_o1 = tf.reshape(q_o1, new_shape=(1, 1, config.n_heads, config.rope_half, 1))
            q_interleaved = tf.concat(q_o0, q_o1, axis=-1)
            q_rope_out = tf.reshape(q_interleaved, new_shape=(1, 1, config.n_heads, config.rope_dim))
            q_final = tf.concat(q_nope, q_rope_out, axis=-1)

            # MQA repeat_interleave to n_heads, for the cache and the new token
            # alike; the KV latent serves as both K and V (no separate V projection).
            k_ctx = tf.cast(
                tf.reshape(
                    tf.transpose(
                        tf.repeat_interleave(kv_cache, repeats=config.n_heads, axis=2),
                        perm=(0, 2, 1, 3),
                    ),
                    new_shape=(1, 1, config.n_heads, C, config.head_dim),
                ),
                dtype="f32",
            )
            k_new = tf.cast(
                tf.repeat_interleave(kv_new, repeats=config.n_heads, axis=2), dtype="f32"
            )
            q_s = tf.cast(q_final * scale, dtype="f32")

            # Two score groups -- one over the cache, one over the token itself --
            # plus the sink's denominator-only column, merged by log-sum-exp
            # against their joint max.
            q_e = tf.reshape(q_s, new_shape=(1, 1, config.n_heads, 1, config.head_dim))
            score_ctx = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum")
            score_new = tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum")
            peak = tf.max(
                tf.max(
                    tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
                ),
                attn_sink,
            )
            peak_e = tf.reshape(peak, new_shape=(1, 1, config.n_heads, 1, 1))
            p_ctx = tf.exp(score_ctx - peak_e)
            p_new = tf.exp(score_new - peak)
            p_sink = tf.exp(attn_sink - peak)
            total = (
                tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum") + p_new + p_sink
            )
            weighted = (
                tf.reduce(p_ctx * k_ctx, axes=(-2,), keepdim=False, kind="sum")
                + p_new * k_new
            )
            ctx = tf.cast(weighted / total, dtype="bf16")

            # Inverse-RoPE: conjugate angle (signs flipped vs. the forward rotation above).
            ctx_nope = ctx[:, :, :, : config.nope_dim]
            ctx_rope_in = ctx[:, :, :, config.nope_dim : config.head_dim]
            ctx_r0 = ctx_rope_in[:, :, :, 0 : config.rope_dim : 2]
            ctx_r1 = ctx_rope_in[:, :, :, 1 : config.rope_dim : 2]
            ctx_r0_f32 = tf.cast(ctx_r0, dtype="f32")
            ctx_r1_f32 = tf.cast(ctx_r1, dtype="f32")
            ctx_o0_f32 = ctx_r0_f32 * cos_pos + ctx_r1_f32 * sin_pos
            ctx_o1_f32 = ctx_r1_f32 * cos_pos - ctx_r0_f32 * sin_pos
            ctx_o0 = tf.cast(ctx_o0_f32, dtype="bf16")
            ctx_o1 = tf.cast(ctx_o1_f32, dtype="bf16")
            ctx_o0 = tf.reshape(ctx_o0, new_shape=(1, 1, config.n_heads, config.rope_half, 1))
            ctx_o1 = tf.reshape(ctx_o1, new_shape=(1, 1, config.n_heads, config.rope_half, 1))
            ctx_interleaved = tf.concat(ctx_o0, ctx_o1, axis=-1)
            ctx_rope_out = tf.reshape(
                ctx_interleaved, new_shape=(1, 1, config.n_heads, config.rope_dim),
            )
            ctx_final = tf.concat(ctx_nope, ctx_rope_out, axis=-1)
            o_flat = tf.reshape(ctx_final, new_shape=(1, 1, config.q_proj))

            # Grouped low-rank O projection: o_flat's last axis is a contiguous
            # o_groups*wo_a_in run, reshaped to (o_groups, 1, 1, wo_a_in) and
            # batched over o_groups -- a @func body can't use a Python loop
            # (it becomes a real IR loop region, not a same-shape unroll).
            o_grouped = tf.reshape(o_flat, new_shape=(config.o_groups, 1, 1, config.wo_a_in))
            w_o_a_grouped = tf.reshape(
                w_o_a, new_shape=(config.o_groups, 1, config.wo_a_in, config.o_lora_rank),
            )
            y_grouped = tf.matmul(o_grouped, w_o_a_grouped)
            y = tf.reshape(y_grouped, new_shape=(1, 1, config.wo_a_out))
            return tf.matmul(y, w_o_b)

        @mla_attend.converter("w_q_a")
        def _(
            wq_a_weight: ConstTensor[(config.q_lora_rank, config.dim), "fp8e4m3"],
            wq_a_scale: ConstTensor[(config.blocks(config.q_lora_rank), config.blocks(config.dim)), "f32"],
        ) -> Tensor[(config.dim, config.q_lora_rank), "bf16"]:
            # Block dequant + transpose, same pattern as w_kv's converter above.
            blocks = tf.reshape(
                tf.cast(wq_a_weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.q_lora_rank), config.quant_block,
                    config.blocks(config.dim), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(wq_a_scale, dtype="bf16"),
                new_shape=(config.blocks(config.q_lora_rank), 1, config.blocks(config.dim), 1),
            )
            dequant = tf.reshape(
                blocks * block_scale, new_shape=(config.q_lora_rank, config.dim),
            )
            return tf.transpose(dequant, perm=(1, 0))

        @mla_attend.converter("w_q_b")
        def _(
            wq_b_weight: ConstTensor[(config.q_proj, config.q_lora_rank), "fp8e4m3"],
            wq_b_scale: ConstTensor[
                (config.blocks(config.q_proj), config.blocks(config.q_lora_rank)), "f32",
            ],
        ) -> Tensor[(config.q_lora_rank, config.q_proj), "bf16"]:
            # Block dequant + transpose, same pattern as w_kv's converter above.
            blocks = tf.reshape(
                tf.cast(wq_b_weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.q_proj), config.quant_block,
                    config.blocks(config.q_lora_rank), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(wq_b_scale, dtype="bf16"),
                new_shape=(config.blocks(config.q_proj), 1, config.blocks(config.q_lora_rank), 1),
            )
            dequant = tf.reshape(
                blocks * block_scale, new_shape=(config.q_proj, config.q_lora_rank),
            )
            return tf.transpose(dequant, perm=(1, 0))

        @mla_attend.converter("attn_sink")
        def _(
            attn_sink_raw: ConstTensor[(config.n_heads,), "f32"],
        ) -> Tensor[(1, 1, config.n_heads, 1), "f32"]:
            # Per-head scalar logit, reshaped to broadcast against the per-head
            # partials it is merged with; f32, which is what the merge runs in.
            return tf.reshape(attn_sink_raw, new_shape=(1, 1, config.n_heads, 1))

        @mla_attend.converter("w_o_a")
        def _(
            wo_a_weight: ConstTensor[(config.wo_a_out, config.dim), "bf16"],
        ) -> Tensor[(config.o_groups, config.wo_a_in, config.o_lora_rank), "bf16"]:
            # Already bf16 (no scale param). Raw weight is contiguous
            # [o_groups*o_lora_rank, wo_a_in]; reshape then transpose to
            # (o_groups, wo_a_in, o_lora_rank) for mla_attend's grouped matmul.
            grouped = tf.reshape(
                wo_a_weight, new_shape=(config.o_groups, config.o_lora_rank, config.wo_a_in),
            )
            return tf.transpose(grouped, perm=(0, 2, 1))

        @mla_attend.converter("w_o_b")
        def _(
            wo_b_weight: ConstTensor[(config.dim, config.wo_a_out), "fp8e4m3"],
            wo_b_scale: ConstTensor[(config.blocks(config.dim), config.blocks(config.wo_a_out)), "f32"],
        ) -> Tensor[(config.wo_a_out, config.dim), "bf16"]:
            # Block dequant + transpose, same pattern as w_kv's converter above.
            blocks = tf.reshape(
                tf.cast(wo_b_weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.dim), config.quant_block,
                    config.blocks(config.wo_a_out), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(wo_b_scale, dtype="bf16"),
                new_shape=(config.blocks(config.dim), 1, config.blocks(config.wo_a_out), 1),
            )
            dequant = tf.reshape(
                blocks * block_scale, new_shape=(config.dim, config.wo_a_out),
            )
            return tf.transpose(dequant, perm=(1, 0))

        def forward(self, hidden, cos_pos, sin_pos, kv_cache, scale, ones_head_dim):
            """Decode-step attention: ``mla_kv_update`` then ``mla_attend``.

            *kv_cache* is the ``ctx_len`` positions before this token, read-only.
            What comes back beside the output is this token's own one-position KV
            latent, for the caller to append.
            """
            kv_new = self.mla_kv_update(hidden, cos_pos, sin_pos)
            out = self.mla_attend(
                hidden, ones_head_dim, cos_pos, sin_pos, kv_cache, kv_new, scale,
            )
            return out, kv_new

    dim = config.dim  # bare-Name locals for the two where(layout=...) spots
    n_act = config.n_act  # below only -- see the module docstring.


    @module(entry="deepseek_v4_flash_moe_hash")
    class DeepseekV4MoE:
        """Hash-router MoE (entry ``deepseek_v4_flash_moe_hash``), plus a plain
        ``forward``. Takes an already-normalized hidden state: the checkpoint's
        ``ffn_norm.weight`` is layer-level, so this component has no pre-MoE norm
        of its own (contrast ``DeepseekV4NoauxTcMoE`` below)."""
        topologies = (Topology("cta", 132), Topology("thread", 512))


        @func
        def shared_fp8_dequant_w1(
            weight: Tensor[(config.moe_inter, config.dim), "fp8e4m3"],
            scale: Tensor[(config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"],
        ) -> Tensor[(config.moe_inter, config.dim), "bf16"]:
            blocks = tf.reshape(
                tf.cast(weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.moe_inter), config.quant_block,
                    config.blocks(config.dim), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(scale, dtype="bf16"),
                new_shape=(config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1),
            )
            return tf.reshape(blocks * block_scale, new_shape=(config.moe_inter, config.dim))

        @func
        def shared_fp8_dequant_w2(
            weight: Tensor[(config.dim, config.moe_inter), "fp8e4m3"],
            scale: Tensor[(config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"],
        ) -> Tensor[(config.dim, config.moe_inter), "bf16"]:
            blocks = tf.reshape(
                tf.cast(weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.dim), config.quant_block,
                    config.blocks(config.moe_inter), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(scale, dtype="bf16"),
                new_shape=(config.blocks(config.dim), 1, config.blocks(config.moe_inter), 1),
            )
            return tf.reshape(blocks * block_scale, new_shape=(config.dim, config.moe_inter))

        @func
        def moe_experts_core(
            x: Tensor[(1, 1, config.dim), "bf16"],
            gweights: Tensor[(1, config.n_act), "f32"],
            eids: Tensor[(1, config.n_act), "i64"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, config.n_act, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))

            gathered_w1 = tf.cast(tf.gather(w1_weight, eids, axis=0), dtype="bf16")
            gathered_s1 = tf.cast(tf.gather(w1_scale, eids, axis=0), dtype="bf16")
            w1 = tf.reshape(
                tf.reshape(
                    gathered_w1,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.moe_inter), config.quant_block,
                        config.blocks(config.dim), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s1,
                    new_shape=(
                        1, config.n_act, config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.moe_inter, config.dim),
            )

            gathered_w3 = tf.cast(tf.gather(w3_weight, eids, axis=0), dtype="bf16")
            gathered_s3 = tf.cast(tf.gather(w3_scale, eids, axis=0), dtype="bf16")
            w3 = tf.reshape(
                tf.reshape(
                    gathered_w3,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.moe_inter), config.quant_block,
                        config.blocks(config.dim), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s3,
                    new_shape=(
                        1, config.n_act, config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.moe_inter, config.dim),
            )

            gathered_w2 = tf.cast(tf.gather(w2_weight, eids, axis=0), dtype="bf16")
            gathered_s2 = tf.cast(tf.gather(w2_scale, eids, axis=0), dtype="bf16")
            w2 = tf.reshape(
                tf.reshape(
                    gathered_w2,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.dim), config.quant_block,
                        config.blocks(config.moe_inter), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s2,
                    new_shape=(
                        1, config.n_act, config.blocks(config.dim), 1, config.blocks(config.moe_inter), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.dim, config.moe_inter),
            )

            token = tf.reshape(xt, new_shape=(1, 1, config.dim, 1))
            gate_value = tf.cast(
                tf.reshape(tf.matmul(w1, token), new_shape=(1, config.n_act, config.moe_inter)),
                dtype="f32",
            )
            up_value = tf.cast(
                tf.reshape(tf.matmul(w3, token), new_shape=(1, config.n_act, config.moe_inter)),
                dtype="f32",
            )
            limit = tf.full_like(up_value, value=config.swiglu_limit)
            up_value = tf.maximum(
                tf.minimum(up_value, limit),
                tf.full_like(up_value, value=-config.swiglu_limit),
            )
            gate_value = tf.minimum(gate_value, limit)
            hidden = (gate_value * tf.sigmoid(gate_value)) * up_value
            hidden = tf.reshape(
                tf.cast(hidden, dtype="bf16"),
                new_shape=(1, config.n_act, config.moe_inter, 1),
            )
            expert_output = tf.cast(
                tf.reshape(tf.matmul(w2, hidden), new_shape=(1, config.n_act, config.dim)),
                dtype="f32",
            )
            weighted = expert_output * tf.reshape(gweights, new_shape=(1, config.n_act, 1))
            return tf.cast(weighted, dtype="bf16")

        @moe_experts_core.converter("w1_scale")
        def _(
            w1_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(w1_scale_raw, dtype="f8e8m0")

        @moe_experts_core.converter("w3_scale")
        def _(
            w3_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(w3_scale_raw, dtype="f8e8m0")

        @moe_experts_core.converter("w2_scale")
        def _(
            w2_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f32"
            ],
        ):
            return tf.cast(w2_scale_raw, dtype="f8e8m0")

        @func
        def moe_hash_gather(
            x: Tensor[(1, 1, config.dim), "bf16"],
            gate_weight: ConstTensor[(config.n_routed, config.dim), "bf16"],
            tid2eid: ConstTensor[(config.vocab, config.n_act), "i64"],
            token_ids: Tensor[(1,), "i64"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, config.n_act, config.dim), "bf16"]:
            # Hash routing: expert ids come from a per-token-id table lookup
            # (tid2eid[token_ids]), not a learned top-k selection, and no bias
            # is added before the gather.
            #
            # tid2eid is stored i64 on disk despite being declared int32 in
            # the reference model -- loaded as i64 directly, matching
            # moe_experts_core's own eids parameter.
            xt = tf.reshape(x, new_shape=(1, config.dim))
            gate = tf.matmul(
                tf.cast(xt, dtype="f32"),
                tf.transpose(tf.cast(gate_weight, dtype="f32"), perm=(1, 0)),
            )
            softplus = tf.log(tf.exp(gate) + tf.full_like(gate, value=1.0))
            scores = softplus * tf.rsqrt(softplus)
            eids = tf.gather(tid2eid, token_ids, axis=0)
            gweights = tf.gather(scores, eids, axis=1, batch_dims=1)
            weight_sum = tf.reduce(
                gweights, axes=(-1,), keepdim=True, kind=ReduceKind.SUM
            )
            gweights = (gweights / weight_sum) * tf.full_like(
                gweights, value=config.route_scale
            )
            return moe_experts_core(
                x, gweights, eids,
                w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            )

        @func
        def shared_expert(
            x: Tensor[(1, 1, config.dim), "bf16"],
            shared_w1_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w1_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w3_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w3_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w2_weight: ConstTensor[(config.dim, config.moe_inter), "fp8e4m3"],
            shared_w2_scale: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))
            w1 = shared_fp8_dequant_w1(shared_w1_weight, shared_w1_scale)
            w3 = shared_fp8_dequant_w1(shared_w3_weight, shared_w3_scale)
            gate = tf.cast(
                tf.matmul(xt, tf.transpose(w1, perm=(1, 0))), dtype="f32"
            )
            up = tf.cast(
                tf.matmul(xt, tf.transpose(w3, perm=(1, 0))), dtype="f32"
            )
            limit = tf.full_like(up, value=config.swiglu_limit)
            up = tf.maximum(
                tf.minimum(up, limit), tf.full_like(up, value=-config.swiglu_limit)
            )
            gate = tf.minimum(gate, limit)
            hidden = tf.cast((gate * tf.sigmoid(gate)) * up, dtype="bf16")
            w2 = shared_fp8_dequant_w2(shared_w2_weight, shared_w2_scale)
            output = tf.cast(
                tf.matmul(hidden, tf.transpose(w2, perm=(1, 0))), dtype="bf16"
            )
            return tf.reshape(output, new_shape=(1, 1, config.dim))

        @shared_expert.converter("shared_w1_scale")
        def _(
            shared_w1_scale_raw: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(shared_w1_scale_raw, dtype="f8e8m0")

        @shared_expert.converter("shared_w3_scale")
        def _(
            shared_w3_scale_raw: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(shared_w3_scale_raw, dtype="f8e8m0")

        @shared_expert.converter("shared_w2_scale")
        def _(
            shared_w2_scale_raw: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f32"
            ],
        ):
            return tf.cast(shared_w2_scale_raw, dtype="f8e8m0")

        @func
        def combine_expert_outputs(
            routed: Tensor[(1, 1, config.dim), "bf16"],
            shared: Tensor[(1, 1, config.dim), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return routed + shared

        @func
        def deepseek_v4_flash_moe_hash(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            gate_weight: ConstTensor[(config.n_routed, config.dim), "bf16"],
            tid2eid: ConstTensor[(config.vocab, config.n_act), "i64"],
            token_ids: Tensor[(1,), "i64"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
            shared_w1_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w1_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w3_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w3_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w2_weight: ConstTensor[(config.dim, config.moe_inter), "fp8e4m3"],
            shared_w2_scale: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            routed_experts: where(layout=(_, n_act @ cta, dim)) = moe_hash_gather(
                hidden, gate_weight, tid2eid, token_ids,
                w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            )
            routed_reduced = tf.reduce(
                routed_experts, axes=(1,), keepdim=False, kind=ReduceKind.SUM,
            )
            routed_value = tf.reshape(
                tf.cast(routed_reduced, dtype="bf16"), new_shape=(1, 1, config.dim),
            )
            shared_value = shared_expert(
                hidden, shared_w1_weight, shared_w1_scale,
                shared_w3_weight, shared_w3_scale, shared_w2_weight, shared_w2_scale,
            )
            combined: where(layout=((_, _, dim), {cta @ B()})) = combine_expert_outputs(
                routed_value, shared_value,
            )
            return combined

        def forward(self, hidden, token_ids):
            """Hash-router MoE, end to end."""
            return self.deepseek_v4_flash_moe_hash(hidden, token_ids)


    @module(entry="deepseek_v4_flash_moe")
    class DeepseekV4NoauxTcMoE:
        """The learned/``noaux_tc``-router MoE component (entry
        ``deepseek_v4_flash_moe``): keeps its own ``pre_moe_rms_norm`` (contrast
        ``DeepseekV4MoE`` above)."""
        topologies = (Topology("cta", 132), Topology("thread", 512))


        @func
        def pre_moe_rms_norm(
            x: Tensor[(1, 1, config.dim), "bf16"],
            rms_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(x, rms_weight)

        @func
        def shared_fp8_dequant_w1(
            weight: Tensor[(config.moe_inter, config.dim), "fp8e4m3"],
            scale: Tensor[(config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"],
        ) -> Tensor[(config.moe_inter, config.dim), "bf16"]:
            blocks = tf.reshape(
                tf.cast(weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.moe_inter), config.quant_block,
                    config.blocks(config.dim), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(scale, dtype="bf16"),
                new_shape=(config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1),
            )
            return tf.reshape(blocks * block_scale, new_shape=(config.moe_inter, config.dim))

        @func
        def shared_fp8_dequant_w2(
            weight: Tensor[(config.dim, config.moe_inter), "fp8e4m3"],
            scale: Tensor[(config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"],
        ) -> Tensor[(config.dim, config.moe_inter), "bf16"]:
            blocks = tf.reshape(
                tf.cast(weight, dtype="bf16"),
                new_shape=(
                    config.blocks(config.dim), config.quant_block,
                    config.blocks(config.moe_inter), config.quant_block,
                ),
            )
            block_scale = tf.reshape(
                tf.cast(scale, dtype="bf16"),
                new_shape=(config.blocks(config.dim), 1, config.blocks(config.moe_inter), 1),
            )
            return tf.reshape(blocks * block_scale, new_shape=(config.dim, config.moe_inter))

        @func
        def moe_experts_core(
            x: Tensor[(1, 1, config.dim), "bf16"],
            gweights: Tensor[(1, config.n_act), "f32"],
            eids: Tensor[(1, config.n_act), "i64"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, config.n_act, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))

            gathered_w1 = tf.cast(tf.gather(w1_weight, eids, axis=0), dtype="bf16")
            gathered_s1 = tf.cast(tf.gather(w1_scale, eids, axis=0), dtype="bf16")
            w1 = tf.reshape(
                tf.reshape(
                    gathered_w1,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.moe_inter), config.quant_block,
                        config.blocks(config.dim), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s1,
                    new_shape=(
                        1, config.n_act, config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.moe_inter, config.dim),
            )

            gathered_w3 = tf.cast(tf.gather(w3_weight, eids, axis=0), dtype="bf16")
            gathered_s3 = tf.cast(tf.gather(w3_scale, eids, axis=0), dtype="bf16")
            w3 = tf.reshape(
                tf.reshape(
                    gathered_w3,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.moe_inter), config.quant_block,
                        config.blocks(config.dim), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s3,
                    new_shape=(
                        1, config.n_act, config.blocks(config.moe_inter), 1, config.blocks(config.dim), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.moe_inter, config.dim),
            )

            gathered_w2 = tf.cast(tf.gather(w2_weight, eids, axis=0), dtype="bf16")
            gathered_s2 = tf.cast(tf.gather(w2_scale, eids, axis=0), dtype="bf16")
            w2 = tf.reshape(
                tf.reshape(
                    gathered_w2,
                    new_shape=(
                        1, config.n_act,
                        config.blocks(config.dim), config.quant_block,
                        config.blocks(config.moe_inter), config.quant_block,
                    ),
                )
                * tf.reshape(
                    gathered_s2,
                    new_shape=(
                        1, config.n_act, config.blocks(config.dim), 1, config.blocks(config.moe_inter), 1
                    ),
                ),
                new_shape=(1, config.n_act, config.dim, config.moe_inter),
            )

            token = tf.reshape(xt, new_shape=(1, 1, config.dim, 1))
            gate_value = tf.cast(
                tf.reshape(tf.matmul(w1, token), new_shape=(1, config.n_act, config.moe_inter)),
                dtype="f32",
            )
            up_value = tf.cast(
                tf.reshape(tf.matmul(w3, token), new_shape=(1, config.n_act, config.moe_inter)),
                dtype="f32",
            )
            limit = tf.full_like(up_value, value=config.swiglu_limit)
            up_value = tf.maximum(
                tf.minimum(up_value, limit),
                tf.full_like(up_value, value=-config.swiglu_limit),
            )
            gate_value = tf.minimum(gate_value, limit)
            hidden = (gate_value * tf.sigmoid(gate_value)) * up_value
            hidden = tf.reshape(
                tf.cast(hidden, dtype="bf16"),
                new_shape=(1, config.n_act, config.moe_inter, 1),
            )
            expert_output = tf.cast(
                tf.reshape(tf.matmul(w2, hidden), new_shape=(1, config.n_act, config.dim)),
                dtype="f32",
            )
            weighted = expert_output * tf.reshape(gweights, new_shape=(1, config.n_act, 1))
            return tf.cast(weighted, dtype="bf16")

        @moe_experts_core.converter("w1_scale")
        def _(
            w1_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(w1_scale_raw, dtype="f8e8m0")

        @moe_experts_core.converter("w3_scale")
        def _(
            w3_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(w3_scale_raw, dtype="f8e8m0")

        @moe_experts_core.converter("w2_scale")
        def _(
            w2_scale_raw: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f32"
            ],
        ):
            return tf.cast(w2_scale_raw, dtype="f8e8m0")

        @func
        def moe_topk(
            x: Tensor[(1, 1, config.dim), "bf16"],
            gate_weight: ConstTensor[(config.n_routed, config.dim), "bf16"],
            gate_bias: ConstTensor[(config.n_routed,), "f32"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, config.n_act, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))
            gate = tf.matmul(
                tf.cast(xt, dtype="f32"),
                tf.transpose(tf.cast(gate_weight, dtype="f32"), perm=(1, 0)),
            )
            softplus = tf.log(tf.exp(gate) + tf.full_like(gate, value=1.0))
            scores = softplus * tf.rsqrt(softplus)
            selection = scores + tf.reshape(gate_bias, new_shape=(1, config.n_routed))
            _, eids = tf.topk(selection, k=config.n_act, axis=-1)
            gweights = tf.gather(scores, eids, axis=1, batch_dims=1)
            weight_sum = tf.reduce(
                gweights, axes=(-1,), keepdim=True, kind=ReduceKind.SUM
            )
            gweights = (gweights / weight_sum) * tf.full_like(
                gweights, value=config.route_scale
            )
            return moe_experts_core(
                x, gweights, eids,
                w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            )

        @func
        def shared_expert(
            x: Tensor[(1, 1, config.dim), "bf16"],
            shared_w1_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w1_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w3_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w3_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w2_weight: ConstTensor[(config.dim, config.moe_inter), "fp8e4m3"],
            shared_w2_scale: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            xt = tf.reshape(x, new_shape=(1, config.dim))
            w1 = shared_fp8_dequant_w1(shared_w1_weight, shared_w1_scale)
            w3 = shared_fp8_dequant_w1(shared_w3_weight, shared_w3_scale)
            gate = tf.cast(
                tf.matmul(xt, tf.transpose(w1, perm=(1, 0))), dtype="f32"
            )
            up = tf.cast(
                tf.matmul(xt, tf.transpose(w3, perm=(1, 0))), dtype="f32"
            )
            limit = tf.full_like(up, value=config.swiglu_limit)
            up = tf.maximum(
                tf.minimum(up, limit), tf.full_like(up, value=-config.swiglu_limit)
            )
            gate = tf.minimum(gate, limit)
            hidden = tf.cast((gate * tf.sigmoid(gate)) * up, dtype="bf16")
            w2 = shared_fp8_dequant_w2(shared_w2_weight, shared_w2_scale)
            output = tf.cast(
                tf.matmul(hidden, tf.transpose(w2, perm=(1, 0))), dtype="bf16"
            )
            return tf.reshape(output, new_shape=(1, 1, config.dim))

        @shared_expert.converter("shared_w1_scale")
        def _(
            shared_w1_scale_raw: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(shared_w1_scale_raw, dtype="f8e8m0")

        @shared_expert.converter("shared_w3_scale")
        def _(
            shared_w3_scale_raw: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f32"
            ],
        ):
            return tf.cast(shared_w3_scale_raw, dtype="f8e8m0")

        @shared_expert.converter("shared_w2_scale")
        def _(
            shared_w2_scale_raw: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f32"
            ],
        ):
            return tf.cast(shared_w2_scale_raw, dtype="f8e8m0")

        @func
        def combine_expert_outputs(
            routed: Tensor[(1, 1, config.dim), "bf16"],
            shared: Tensor[(1, 1, config.dim), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return routed + shared

        @func
        def deepseek_v4_flash_moe(
            x: Tensor[(1, 1, config.dim), "bf16"],
            rms_weight: ConstTensor[(config.dim,), "bf16"],
            gate_weight: ConstTensor[(config.n_routed, config.dim), "bf16"],
            gate_bias: ConstTensor[(config.n_routed,), "f32"],
            w1_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w1_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w3_weight: ConstTensor[(config.n_routed, config.moe_inter, config.dim), "fp8e4m3"],
            w3_scale: ConstTensor[
                (config.n_routed, config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            w2_weight: ConstTensor[(config.n_routed, config.dim, config.moe_inter), "fp8e4m3"],
            w2_scale: ConstTensor[
                (config.n_routed, config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
            shared_w1_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w1_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w3_weight: ConstTensor[(config.moe_inter, config.dim), "fp8e4m3"],
            shared_w3_scale: ConstTensor[
                (config.blocks(config.moe_inter), config.blocks(config.dim)), "f8e8m0"
            ],
            shared_w2_weight: ConstTensor[(config.dim, config.moe_inter), "fp8e4m3"],
            shared_w2_scale: ConstTensor[
                (config.blocks(config.dim), config.blocks(config.moe_inter)), "f8e8m0"
            ],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            hidden = pre_moe_rms_norm(x, rms_weight)
            routed_experts: where(layout=(_, n_act @ cta, dim)) = moe_topk(
                hidden, gate_weight, gate_bias,
                w1_weight, w1_scale, w3_weight, w3_scale, w2_weight, w2_scale,
            )
            routed_reduced = tf.reduce(
                routed_experts, axes=(1,), keepdim=False, kind=ReduceKind.SUM,
            )
            routed_value = tf.reshape(
                tf.cast(routed_reduced, dtype="bf16"), new_shape=(1, 1, config.dim),
            )
            shared_value = shared_expert(
                hidden, shared_w1_weight, shared_w1_scale,
                shared_w3_weight, shared_w3_scale, shared_w2_weight, shared_w2_scale,
            )
            combined: where(layout=((_, _, dim), {cta @ B()})) = combine_expert_outputs(
                routed_value, shared_value,
            )
            return combined

    return DeepseekV4Attention, DeepseekV4MoE, DeepseekV4NoauxTcMoE


def build_deepseek_v4_flash(config: DSV4Config):
    """The whole tree at *config*: embedding, ``n_layers`` decoder layers, final
    norm, head, and the decode hooks a generation loop calls.

    Public because this model is asked about more than one shape, and a
    ``@module`` class body at file scope is evaluated once. Callers name the
    shape they mean -- ``REAL`` here, or the small shape a test builds -- and get a tree
    that shares no IR node with any other call's.
    """
    attention_module, moe_module, _ = _submodules(config)

    @module(entry="residual_add")
    class DeepseekV4DecoderLayer:
        @func
        def pre_attn_rms_norm(
            x: Tensor[(1, 1, config.dim), "bf16"],
            pre_attn_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(x, pre_attn_norm_weight)

        @func
        def pre_moe_rms_norm(
            x: Tensor[(1, 1, config.dim), "bf16"],
            pre_moe_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            # ffn_norm.weight is a layer-level tensor (real checkpoint), not part of moe.
            return tf.rms_norm(x, pre_moe_norm_weight)

        @func
        def residual_add(
            a: Tensor[(1, 1, config.dim), "bf16"],
            b: Tensor[(1, 1, config.dim), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return a + b

        attention = attention_module
        moe = moe_module

        def forward(self, hidden, cos_pos, sin_pos, kv_cache, scale, ones_head_dim, token_ids):
            attn_in = self.pre_attn_rms_norm(hidden)
            attn_out, kv_new = self.attention(
                attn_in, cos_pos, sin_pos, kv_cache, scale, ones_head_dim,
            )
            h1 = self.residual_add(hidden, attn_out)
            moe_in = self.pre_moe_rms_norm(h1)
            moe_out = self.moe(moe_in, token_ids)
            out = self.residual_add(h1, moe_out)
            return out, kv_new

    #: How many positions the caller may keep: a query attends ``window``
    #: positions counting its own, so the context it is handed is one shorter.
    MAX_CTX = config.window - 1

    @module(entry="lm_head", target=CudaTarget("nvidia.h200_sxm"))
    class DeepseekV4ForCausalLM:
        topologies = (Topology("cta", 132), Topology("thread", 512))

        @func
        def embed(
            table: ConstTensor[(config.vocab, config.dim), "bf16"],
            token_ids: Tensor[(1,), "i64"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.reshape(tf.gather(table, token_ids, axis=0), new_shape=(1, 1, config.dim))

        @func
        def final_rms_norm(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            final_norm_weight: ConstTensor[(config.dim,), "bf16"],
        ) -> Tensor[(1, 1, config.dim), "bf16"]:
            return tf.rms_norm(hidden, final_norm_weight)

        @func
        def lm_head(
            hidden: Tensor[(1, 1, config.dim), "bf16"],
            lm_head_weight: ConstTensor[(config.dim, config.vocab), "bf16"],
        ) -> Tensor[(1, 1, config.vocab), "bf16"]:
            logits = tf.matmul(tf.reshape(hidden, new_shape=(1, config.dim)), lm_head_weight)
            return tf.reshape(logits, new_shape=(1, 1, config.vocab))

        @lm_head.converter("lm_head_weight")
        def _(
            head_weight_raw: ConstTensor[(config.vocab, config.dim), "bf16"],
        ) -> Tensor[(config.dim, config.vocab), "bf16"]:
            # head.weight is (vocab, dim); transpose to match lm_head's (dim, vocab) matmul.
            return tf.transpose(head_weight_raw, perm=(1, 0))

        layers = tuple(
            DeepseekV4DecoderLayer.renamed(f"layer{index}")
            for index in range(config.n_layers)
        )

        def forward(self, token_ids, cos_pos, sin_pos, past_key_values, scale, ones_head_dim):
            """One decode step of the whole model: the per-layer context it reads
            in, and each layer's own one-position KV latent out. Growing the
            context with those is ``append_cache``, and the caller's step.
            """
            hidden = self.embed(token_ids)
            fresh = []
            for i in range(config.n_layers):
                layer = getattr(self, f"layer{i}")
                hidden, kv_new = layer(
                    hidden, cos_pos, sin_pos, past_key_values[i], scale, ones_head_dim, token_ids,
                )
                fresh.append(kv_new)
            normed = self.final_rms_norm(hidden)
            logits = self.lm_head(normed)
            return logits, tuple(fresh)

        def append_cache(self, caches, fresh):
            """The context the next step reads: each layer's fresh one-position KV
            latent appended to the cache it was given, and what falls out of the
            window dropped. A policy over tensors the caller holds, which is what
            lets every shape below be expressed in ``ctx_len`` alone.
            """
            import torch  # noqa: PLC0415

            appended = []
            for cache, kv_new in zip(caches, fresh):
                grown = torch.cat([cache, kv_new], dim=1)
                appended.append(grown[:, -MAX_CTX:] if grown.shape[1] > MAX_CTX else grown)
            return tuple(appended)

        def init_caches(self, device="cuda", mesh=None):
            """The per-layer context the decode loop starts from, drawn at a fixed
            seed so every caller of this model starts from the same one."""
            import torch  # noqa: PLC0415

            generator = torch.Generator(device=device).manual_seed(SEED_CTX_SEED)
            return tuple(
                (
                    torch.randn(
                        1, SEED_CTX_LEN, 1, config.head_dim,
                        generator=generator, device=device, dtype=torch.float32,
                    )
                    * 0.1
                ).to(torch.bfloat16)
                for _ in range(config.n_layers)
            )

        def prepare_inputs_for_generation(self, input_ids, step, past_key_values, device="cuda"):
            import torch  # noqa: PLC0415

            ids = input_ids.reshape(-1)
            token_ids = ids[step].reshape(1).to(device=device, dtype=torch.int64)
            # The step's own absolute position: the seed context occupies the ones
            # before it, and rotation is what ties a key to the position it was
            # written at.
            cos, sin = _rope_cos_sin(SEED_CTX_LEN + step, device=device)
            cos_pos = cos.view(1, 1, 1, config.rope_half)
            sin_pos = sin.view(1, 1, 1, config.rope_half)
            scale = torch.full(
                (1, 1, 1, 1), config.head_dim ** -0.5, device=device, dtype=torch.bfloat16,
            )
            ones_head_dim = torch.ones(config.head_dim, device=device, dtype=torch.bfloat16)
            return (token_ids, cos_pos, sin_pos, past_key_values, scale, ones_head_dim)

    return DeepseekV4ForCausalLM


#: The attention submodule and the ``noaux_tc`` MoE block at the real checkpoint's
#: dimensions, built on their own for the tests that ask about a block rather than
#: about the model. Neither declares a Target.
DeepseekV4Attention, _, deepseek_v4_flash_module = _submodules(REAL)

#: The published model, at the real checkpoint's shape. Any other shape is the
#: caller's to name: ``build_deepseek_v4_flash(TINY)``.
DeepseekV4ForCausalLM = build_deepseek_v4_flash(REAL)

__all__ = [
    "SEED_CTX_LEN",
    "SEED_CTX_SEED",
    "DeepseekV4Attention",
    "DeepseekV4ForCausalLM",
    "build_deepseek_v4_flash",
    "deepseek_v4_flash_module",
]
