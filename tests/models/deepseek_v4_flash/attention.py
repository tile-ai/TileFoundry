"""DeepSeek-V4-Flash decode-step attention component: real transformer layer 0
(config.json ``compress_ratios[0] == 0``) -- pure sliding-window MLA.

Config-driven: ``build_attention(config: DSV4Config) -> Module`` builds the two
``@func``s below at *config*'s dimensions and wires them into a ``Module``;
``attention_module = build_attention(REAL)`` is the real-scale instance
``decode_step.py`` imports. ``DSV4Config.tiny()`` is the same architecture at
a size an end-to-end numeric/executable check can run (see config.py).

Two chained ``@func``s (the caller makes two separate ``evaluate()`` calls,
not one composed ``@func``): ``mla_kv_update`` projects and RMSNorms the
shared KV latent, applies partial RoPE to its last ``config.rope_dim`` dims,
fp8 fake-quantizes the non-rope portion, and writes the result into a fixed
``config.window``-token cache slot. ``mla_attend`` computes the low-rank Q
projection (``wq_a`` -> RMS rescale -> ``wq_b``), applies the same partial
RoPE, attends the single new-token query over the cached KV latent (MQA:
``n_kv_heads == 1``, the cache serves as both K and V -- MLA-absorbed, no
separate V projection) with an ``attn_sink`` softmax column, inverse-RoPEs
the context, and applies the grouped low-rank O projection (``config.o_groups``
groups, ``wo_a`` per group -> ``wo_b``). ``forward`` chains the two: cache
in, cache out, no hidden state.

Checkpoint-backed params are declared ``ConstTensor`` (the module's derived
``weights``); every other param (``hidden``, ``cos_pos``/``sin_pos``,
``cur_pos``/``s``, ``kv_cache0``/``kv_cache``, ``attn_mask``, ``scale``, and
``ones_head_dim``) stays a plain ``Tensor`` -- not checkpoint data, just
caller-supplied activations, the functional cache, or (``ones_head_dim``) a
fixed all-ones RMSNorm weight. Each ``ConstTensor`` gets at most one
``.converter(...)`` turning the real FP8 checkpoint's raw tensor into the
canonical shape/dtype the ``@func`` body uses: block-dequant + transpose for
the four fp8 projection weights (``w_kv``, ``w_q_a``, ``w_q_b``, ``w_o_b`` --
same op sequence as ``moe.py::shared_fp8_dequant_w1``), a reshape + cast for
``attn_sink``, a reshape + transpose for the un-scaled ``w_o_a``, and no
converter at all (pass-through by name) for the two RMSNorm gammas
(``gamma_kv``, ``gamma_q_lora``). Raw scales are **F32** in the checkpoint
(``scale_fmt: ue8m0`` means the values are powers of two, not that the
storage is f8e8m0). Shapes/dtypes below are verified against the real
``DeepSeek-V4-Flash-FP8`` checkpoint at *config* = ``REAL``.

RoPE here uses DeepSeek's interleaved-pairs convention (view-as-complex on
adjacent dims), distinct from ``tilefoundry.dsl.tf.rope``'s rotate-half
convention, so it is built from ``tf.slice``/``tf.reshape``/``tf.concat``
primitives instead of that op.
"""
from __future__ import annotations

from tests.models.deepseek_v4_flash.config import (
    FP8E4M3_MAX,
    FP8E4M3_QUANT_EPS,
    KV_QUANT_BLOCK,
    REAL,
    DSV4Config,
)
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, ReduceKind, Tensor, tf
from tilefoundry.ir.core.module import Module


def build_attention(config: DSV4Config) -> Module:
    """Build the attention component ``Module`` at *config*'s dimensions.

    Two ``@func``s (``mla_kv_update``, ``mla_attend``) plus their per-weight
    converters and a plain ``forward`` orchestration method (cache in, cache
    out -- see module docstring). Every shape traces to *config*; the KV-cache
    fp8 fake-quant block size and its e4m3 grid are architectural constants
    (``KV_QUANT_BLOCK``/``FP8E4M3_MAX``/``FP8E4M3_QUANT_EPS``, imported from
    config.py), not a per-*config* choice.
    """
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
            # Single shared head_dim-wide KV latent (MQA, n_kv_heads==1): wkv ->
            # kv_norm -> RoPE on the last rope_dim dims (interleaved-pairs
            # convention, inlined -- see note above and
            # hf_attention_ref.apply_rotary_emb) -> functional fixed-capacity
            # cache write.
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

            # Official additionally fake-quantizes the cached KV latent's non-rope
            # portion through an FP8 e4m3 grid with a power-of-2 ("ue8m0") block
            # scale before caching (QAT-noise simulation; hf_attention_ref.
            # _fake_quant_fp8_block / kernel.py's `act_quant(..., inplace=True)`,
            # round_scale=True): reshape into 64-wide blocks, block-absmax -> clamp
            # to a floor -> round the scale up to a power of 2 (exp2(ceil(log2(.)))
            # -- needs the CEIL/EXP2/LOG2 unary ops) -> divide -> clamp to the fp8
            # range -> real fp8e4m3 cast round-trip -> multiply back by the scale.
            # kv_rope_in (the last rope_dim dims) is intentionally left
            # bf16/unquantized, matching the official "rope dims kept for
            # positional precision" comment (model.py).
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

            # f32 upcast for the rotation itself, single rounding back to bf16 at the
            # end -- matches official apply_rotary_emb's x.float() ... y.copy_(x)
            # (see hf_attention_ref.py); cos_pos/sin_pos are f32-typed (see signature)
            # so no separate cast is needed for them.
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
            # Block dequant (weight * scale broadcast over quant_block x
            # quant_block tiles -- same op sequence as
            # moe.py::shared_fp8_dequant_w1: reshape to 4-D tiles, reshape the
            # scale to (b0, 1, b1, 1), multiply, reshape back), then transpose
            # to the canonical (dim, head_dim) orientation ``mla_kv_update``
            # expects.
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
            # Low-rank Q (wq_a -> q_norm -> wq_b), per-head unweighted RMS rescale
            # (official: ``q *= rsqrt(mean(q**2,-1)+eps)``, no learned weight --
            # reproduced via ``tf.rms_norm`` with an all-ones weight; official does
            # this one step without an fp32 upcast, rms_norm's evaluator upcasts
            # internally like its other calls -- a minor, flagged precision-only
            # deviation, see report), RoPE on the last rope_dim dims (inlined, see
            # note above this function).
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
            # f32 upcast for the rotation itself, single rounding back to bf16 (see
            # mla_kv_update's identical rope block for the rationale).
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

            # MQA broadcast (n_kv_heads==1 -> n_heads via repeat_interleave, same
            # op/pattern as this file's own GQA placeholder above and
            # qwen3_module.py); kv_cache serves as both K and V (MLA-absorbed: no
            # separate V projection).
            k_b = tf.repeat_interleave(kv_cache, repeats=config.n_heads, axis=2)
            q_h = tf.transpose(q_final, perm=(0, 2, 1, 3))
            k_h = tf.transpose(k_b, perm=(0, 2, 1, 3))
            q_s = tf.mul(q_h, scale)
            k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
            scores = tf.add(tf.matmul(q_s, k_t), attn_mask)

            # attn_sink: a learned denominator-only logit, folded in as one extra
            # softmax column with no corresponding value (kernel.py's `sparse_attn`;
            # see hf_attention_ref.sparse_attn_torch for the equivalence) -- appended
            # via concat, then sliced back off before the P@V matmul.
            scores_ext = tf.concat(scores, attn_sink, axis=-1)
            probs_ext = tf.softmax(scores_ext, axis=-1)
            probs = tf.slice(
                probs_ext,
                begin=(0, 0, 0, 0),
                end=(1, config.n_heads, 1, config.window),
                strides=(1, 1, 1, 1),
            )
            ctx = tf.matmul(probs, k_h)

            # Inverse-RoPE the attention output's last rope_dim dims (official:
            # ``apply_rotary_emb(o[...,-rd:], freqs_cis, True)``, same query
            # position; inverse uses the conjugate angle: (x0*cos+x1*sin,
            # x1*cos-x0*sin) -- see hf_attention_ref.apply_rotary_emb).
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
            # f32 upcast for the rotation itself, single rounding back to bf16 (see
            # mla_kv_update's identical rope block for the rationale).
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

            # Grouped low-rank O projection (wo_a): official reinterprets one
            # [WO_A_OUT, WO_A_IN] weight as o_groups independent [O_LORA_RANK,
            # WO_A_IN] blocks, each applied to its own consecutive-heads input
            # slice, n_heads/o_groups heads wide (``torch.einsum("bsgd,grd->bsgr",
            # o, wo_a)`` in model.py). o_flat's last axis is a contiguous
            # o_groups*wo_a_in run (group g owns [g*wo_a_in:(g+1)*wo_a_in]), so a
            # plain reshape splits it into (o_groups, 1, 1, wo_a_in) with no data
            # movement; w_o_a is already (o_groups, wo_a_in, o_lora_rank),
            # reshaped the same way to line up a matching batch axis. One
            # N-D-broadcasting batched ``tf.matmul`` over that o_groups axis then
            # reproduces every group's independent matmul in a single call (the
            # same batched-matmul pattern moe.py's moe_experts_core uses over its
            # N_ACT axis -- verified, unlike when this file's o_groups was a
            # fixed literal), and a plain reshape re-flattens (o_groups,
            # o_lora_rank) back to wo_a_out in the same group order a per-group
            # concat would have. This is config-driven (o_groups is
            # o_groups is 8 at REAL, 2 at DSV4Config.tiny()), which
            # rules out unrolling a fixed number of slice+matmul+concat groups by
            # hand -- and a @func body cannot use a Python for loop or a plain
            # helper to do it instead (a for loop becomes a genuine IR loop
            # region with carried-accumulator semantics, not a same-shape
            # unroll; see module docstring).
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
            # A per-head scalar logit, reshaped to broadcast against the
            # (1, n_heads, 1, window) scores this concats onto and cast to bf16.
            reshaped = tf.reshape(attn_sink_raw, new_shape=(1, config.n_heads, 1, 1))
            return tf.cast(reshaped, dtype="bf16")

        @mla_attend.converter("w_o_a")
        def _(
            wo_a_weight: ConstTensor[(config.wo_a_out, config.dim), "bf16"],
        ) -> Tensor[(config.o_groups, config.wo_a_in, config.o_lora_rank), "bf16"]:
            # No scale (raw is already bf16). The raw weight is one contiguous
            # [wo_a_out, dim] == [o_groups*o_lora_rank, wo_a_in] matrix (dim ==
            # wo_a_in for this architecture -- see module docstring / config.py);
            # a plain reshape splits its first axis into (o_groups, o_lora_rank),
            # matching the official [g, r, d] view (model.py's
            # ``torch.einsum("bsgd,grd->bsgr", o, wo_a)``), then the last two
            # axes are transposed to this file's own (o_groups, wo_a_in,
            # o_lora_rank) parameter order (see mla_attend's grouped O-projection
            # comment for how that order is used).
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
            cache in, cache out, no hidden state. ``self.mla_kv_update`` /
            ``self.mla_attend`` are ``Module.__getattr__``'s callables, so weights
            are filled by name from this Module's own bound state (``load``) and
            this method threads only the activations plus the functional KV
            cache -- the same spelling a ``RuntimeModule`` twin answers with a
            kernel, which is what lets this method be reused unchanged there.
            """
            kv_cache1 = self.mla_kv_update(hidden, cos_pos, sin_pos, kv_cache0, cur_pos, s)
            out = self.mla_attend(
                hidden, ones_head_dim, cos_pos, sin_pos, kv_cache1, attn_mask, scale,
            )
            return out, kv_cache1

    return DeepseekV4Attention


attention_module = build_attention(REAL)


__all__ = [
    "attention_module",
    "build_attention",
]
