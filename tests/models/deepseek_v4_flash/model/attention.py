"""DeepSeek-V4-Flash decode-step attention (real layer 0, sliding-window
MLA): ``DeepseekV4Attention`` builds ``mla_kv_update`` and ``mla_attend`` at
*config*'s dimensions (a free name, injected by ``tests.models.loader``).

RoPE uses DeepSeek's interleaved-pairs convention (view-as-complex on
adjacent dims), not ``tilefoundry.dsl.tf.rope``'s rotate-half, so it is built
here from ``tf.slice``/``tf.reshape``/``tf.concat`` primitives instead.
"""
from __future__ import annotations

from tests.models.deepseek_v4_flash.config import (
    FP8E4M3_MAX,
    FP8E4M3_QUANT_EPS,
    KV_QUANT_BLOCK,
)
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, ReduceKind, Tensor, tf


@module(entry="mla_attend")
class DeepseekV4Attention:
    @func
    def mla_kv_update(
        hidden: Tensor[(1, 1, config.dim), "bf16"],
        gamma_kv: ConstTensor[(config.head_dim,), "bf16"],
        w_kv: ConstTensor[(config.dim, config.head_dim), "bf16"],
        cos_pos: Tensor[(1, 1, 1, config.rope_half), "f32"],
        sin_pos: Tensor[(1, 1, 1, config.rope_half), "f32"],
        kv_cache0: Tensor[(1, config.window, 1, config.head_dim), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        s: Tensor[(1,), "i32"],
    ) -> Tensor[(1, config.window, 1, config.head_dim), "bf16"]:
        # MQA: one shared head_dim-wide KV latent (n_kv_heads == 1).
        kv = tf.matmul(hidden, w_kv)
        kv_n = tf.rms_norm(kv, gamma_kv)
        kv_4d = tf.reshape(kv_n, new_shape=(1, 1, 1, config.head_dim))
        kv_nope = tf.slice(
            kv_4d, begin=(0, 0, 0, 0), end=(1, 1, 1, config.nope_dim), strides=(1, 1, 1, 1),
        )
        kv_rope_in = tf.slice(
            kv_4d,
            begin=(0, 0, 0, config.nope_dim),
            end=(1, 1, 1, config.head_dim),
            strides=(1, 1, 1, 1),
        )

        # FP8 e4m3 fake-quant of the non-rope KV latent: block-absmax, scale
        # rounded up to a power of two (ue8m0), then a real fp8e4m3 round
        # trip. kv_rope_in stays bf16/unquantized.
        kv_nope_f32 = tf.cast(kv_nope, dtype="f32")
        kv_nope_blk = tf.reshape(
            kv_nope_f32, new_shape=(1, 1, 1, config.kv_quant_blocks, KV_QUANT_BLOCK),
        )
        kv_amax = tf.reduce(kv_nope_blk, axes=(-1,), keepdim=True, kind=ReduceKind.ABS_MAX)
        kv_amax = tf.max(kv_amax, FP8E4M3_QUANT_EPS)
        kv_scale = tf.exp2(tf.ceil(tf.log2(tf.div(kv_amax, FP8E4M3_MAX))))
        kv_scaled = tf.div(kv_nope_blk, kv_scale)
        kv_scaled = tf.min(tf.max(kv_scaled, -FP8E4M3_MAX), FP8E4M3_MAX)
        kv_q_fp8 = tf.cast(kv_scaled, dtype="fp8e4m3")
        kv_dq = tf.mul(tf.cast(kv_q_fp8, dtype="f32"), kv_scale)
        kv_nope_q = tf.cast(tf.reshape(kv_dq, new_shape=(1, 1, 1, config.nope_dim)), dtype="bf16")

        kv_r0 = tf.slice(
            kv_rope_in, begin=(0, 0, 0, 0), end=(1, 1, 1, config.rope_dim), strides=(1, 1, 1, 2),
        )
        kv_r1 = tf.slice(
            kv_rope_in, begin=(0, 0, 0, 1), end=(1, 1, 1, config.rope_dim), strides=(1, 1, 1, 2),
        )
        kv_r0_f32 = tf.cast(kv_r0, dtype="f32")
        kv_r1_f32 = tf.cast(kv_r1, dtype="f32")
        kv_o0_f32 = tf.sub(tf.mul(kv_r0_f32, cos_pos), tf.mul(kv_r1_f32, sin_pos))
        kv_o1_f32 = tf.add(tf.mul(kv_r0_f32, sin_pos), tf.mul(kv_r1_f32, cos_pos))
        kv_o0 = tf.cast(kv_o0_f32, dtype="bf16")
        kv_o1 = tf.cast(kv_o1_f32, dtype="bf16")
        kv_o0 = tf.reshape(kv_o0, new_shape=(1, 1, 1, config.rope_half, 1))
        kv_o1 = tf.reshape(kv_o1, new_shape=(1, 1, 1, config.rope_half, 1))
        kv_interleaved = tf.concat(kv_o0, kv_o1, axis=-1)
        kv_rope_out = tf.reshape(kv_interleaved, new_shape=(1, 1, 1, config.rope_dim))
        kv_final = tf.concat(kv_nope_q, kv_rope_out, axis=-1)
        return tf.cache_update(kv_cache0, cur_pos, s, kv_final)

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
        dequant = tf.reshape(tf.mul(blocks, block_scale), new_shape=(config.head_dim, config.dim))
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
        kv_cache: Tensor[(1, config.window, 1, config.head_dim), "bf16"],
        attn_mask: Tensor[(1, 1, 1, config.window), "bf16"],
        attn_sink: ConstTensor[(1, config.n_heads, 1, 1), "bf16"],
        scale: Tensor[(1, 1, 1, 1), "bf16"],
        w_o_a: ConstTensor[(config.o_groups, config.wo_a_in, config.o_lora_rank), "bf16"],
        w_o_b: ConstTensor[(config.wo_a_out, config.dim), "bf16"],
    ) -> Tensor[(1, 1, config.dim), "bf16"]:
        # q_rescaled is a per-head unweighted RMS rescale (tf.rms_norm, all-ones weight).
        q_lat = tf.rms_norm(tf.matmul(hidden, w_q_a), gamma_q_lora)
        q_full = tf.matmul(q_lat, w_q_b)
        q = tf.reshape(q_full, new_shape=(1, 1, config.n_heads, config.head_dim))
        q_rescaled = tf.rms_norm(q, ones_head_dim)
        q_nope = tf.slice(
            q_rescaled,
            begin=(0, 0, 0, 0),
            end=(1, 1, config.n_heads, config.nope_dim),
            strides=(1, 1, 1, 1),
        )
        q_rope_in = tf.slice(
            q_rescaled,
            begin=(0, 0, 0, config.nope_dim),
            end=(1, 1, config.n_heads, config.head_dim),
            strides=(1, 1, 1, 1),
        )
        q_r0 = tf.slice(
            q_rope_in,
            begin=(0, 0, 0, 0),
            end=(1, 1, config.n_heads, config.rope_dim),
            strides=(1, 1, 1, 2),
        )
        q_r1 = tf.slice(
            q_rope_in,
            begin=(0, 0, 0, 1),
            end=(1, 1, config.n_heads, config.rope_dim),
            strides=(1, 1, 1, 2),
        )
        q_r0_f32 = tf.cast(q_r0, dtype="f32")
        q_r1_f32 = tf.cast(q_r1, dtype="f32")
        q_o0_f32 = tf.sub(tf.mul(q_r0_f32, cos_pos), tf.mul(q_r1_f32, sin_pos))
        q_o1_f32 = tf.add(tf.mul(q_r0_f32, sin_pos), tf.mul(q_r1_f32, cos_pos))
        q_o0 = tf.cast(q_o0_f32, dtype="bf16")
        q_o1 = tf.cast(q_o1_f32, dtype="bf16")
        q_o0 = tf.reshape(q_o0, new_shape=(1, 1, config.n_heads, config.rope_half, 1))
        q_o1 = tf.reshape(q_o1, new_shape=(1, 1, config.n_heads, config.rope_half, 1))
        q_interleaved = tf.concat(q_o0, q_o1, axis=-1)
        q_rope_out = tf.reshape(q_interleaved, new_shape=(1, 1, config.n_heads, config.rope_dim))
        q_final = tf.concat(q_nope, q_rope_out, axis=-1)

        # MQA repeat_interleave to n_heads; kv_cache serves as both K and V (no separate V projection).
        k_b = tf.repeat_interleave(kv_cache, repeats=config.n_heads, axis=2)
        q_h = tf.transpose(q_final, perm=(0, 2, 1, 3))
        k_h = tf.transpose(k_b, perm=(0, 2, 1, 3))
        q_s = tf.mul(q_h, scale)
        k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
        scores = tf.add(tf.matmul(q_s, k_t), attn_mask)

        # attn_sink: an extra denominator-only softmax column, sliced back off before the P@V matmul.
        scores_ext = tf.concat(scores, attn_sink, axis=-1)
        probs_ext = tf.softmax(scores_ext, axis=-1)
        probs = tf.slice(
            probs_ext,
            begin=(0, 0, 0, 0),
            end=(1, config.n_heads, 1, config.window),
            strides=(1, 1, 1, 1),
        )
        ctx = tf.matmul(probs, k_h)

        # Inverse-RoPE: conjugate angle (signs flipped vs. the forward rotation above).
        ctx_nope = tf.slice(
            ctx,
            begin=(0, 0, 0, 0),
            end=(1, config.n_heads, 1, config.nope_dim),
            strides=(1, 1, 1, 1),
        )
        ctx_rope_in = tf.slice(
            ctx,
            begin=(0, 0, 0, config.nope_dim),
            end=(1, config.n_heads, 1, config.head_dim),
            strides=(1, 1, 1, 1),
        )
        ctx_r0 = tf.slice(
            ctx_rope_in,
            begin=(0, 0, 0, 0),
            end=(1, config.n_heads, 1, config.rope_dim),
            strides=(1, 1, 1, 2),
        )
        ctx_r1 = tf.slice(
            ctx_rope_in,
            begin=(0, 0, 0, 1),
            end=(1, config.n_heads, 1, config.rope_dim),
            strides=(1, 1, 1, 2),
        )
        ctx_r0_f32 = tf.cast(ctx_r0, dtype="f32")
        ctx_r1_f32 = tf.cast(ctx_r1, dtype="f32")
        ctx_o0_f32 = tf.add(tf.mul(ctx_r0_f32, cos_pos), tf.mul(ctx_r1_f32, sin_pos))
        ctx_o1_f32 = tf.sub(tf.mul(ctx_r1_f32, cos_pos), tf.mul(ctx_r0_f32, sin_pos))
        ctx_o0 = tf.cast(ctx_o0_f32, dtype="bf16")
        ctx_o1 = tf.cast(ctx_o1_f32, dtype="bf16")
        ctx_o0 = tf.reshape(ctx_o0, new_shape=(1, config.n_heads, 1, config.rope_half, 1))
        ctx_o1 = tf.reshape(ctx_o1, new_shape=(1, config.n_heads, 1, config.rope_half, 1))
        ctx_interleaved = tf.concat(ctx_o0, ctx_o1, axis=-1)
        ctx_rope_out = tf.reshape(
            ctx_interleaved, new_shape=(1, config.n_heads, 1, config.rope_dim),
        )
        ctx_final = tf.concat(ctx_nope, ctx_rope_out, axis=-1)

        attn_out_heads_last = tf.transpose(ctx_final, perm=(0, 2, 1, 3))
        o_flat = tf.reshape(attn_out_heads_last, new_shape=(1, 1, config.q_proj))

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
            tf.mul(blocks, block_scale), new_shape=(config.q_lora_rank, config.dim),
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
            tf.mul(blocks, block_scale), new_shape=(config.q_proj, config.q_lora_rank),
        )
        return tf.transpose(dequant, perm=(1, 0))

    @mla_attend.converter("attn_sink")
    def _(
        attn_sink_raw: ConstTensor[(config.n_heads,), "f32"],
    ) -> Tensor[(1, config.n_heads, 1, 1), "bf16"]:
        # Per-head scalar logit, reshaped to broadcast against the scores this concats onto.
        reshaped = tf.reshape(attn_sink_raw, new_shape=(1, config.n_heads, 1, 1))
        return tf.cast(reshaped, dtype="bf16")

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
            tf.mul(blocks, block_scale), new_shape=(config.dim, config.wo_a_out),
        )
        return tf.transpose(dequant, perm=(1, 0))

    def forward(
        self, hidden, cos_pos, sin_pos, cur_pos, s, kv_cache0, attn_mask, scale, ones_head_dim,
    ):
        """Decode-step attention: ``mla_kv_update`` then ``mla_attend`` --
        cache in, cache out, no hidden state."""
        kv_cache1 = self.mla_kv_update(hidden, cos_pos, sin_pos, kv_cache0, cur_pos, s)
        out = self.mla_attend(
            hidden, ones_head_dim, cos_pos, sin_pos, kv_cache1, attn_mask, scale,
        )
        return out, kv_cache1
