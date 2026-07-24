"""DeepSeek V4 flash decode-step attention component.

Head configuration is **GQA** (32 query heads / 4 KV heads / 128 head dim) as
a placeholder for the real MLA attention — MLA's own head/latent dims land
once the architecture parameters are confirmed; the leaf boundary this module
exposes (``attn_kv_update`` / ``attn_scores``) does not change either way.

Mirrors ``tests/models/qwen3_5_30b_a3b/common.py``'s ``build_kv_update`` /
``build_scores`` split: a concat-derived compound ``DimVar`` axis (prior +
new context length) cannot feed ``matmul`` / ``transpose`` in the same
Function (the relation engine cannot recover it), so the KV-cache append and
the score computation are two ``@func``s, chained in Python — see
``tests/models/qwen3_5_30b_a3b/test_attention.py``. Unlike qwen's decode step
(``build_decode_attention``, a fixed-capacity ``cache_update`` write), the
context length here is a genuine ``DimVar``: the same built Function
instances evaluate at any concrete prior/total context length without being
rebuilt (docs/plans/agent-kernel-loop/P0a-tonight-nested-module-e2e.md, M2).

Decode regime: batch 1, exactly one new token per step (``S = 1``, matching
``moe.py``'s ``(1, 1, DIM)`` token shape) against a dynamic prior context.
"""
from __future__ import annotations

from tests.models.deepseek_v4_flash.moe import DIM
from tilefoundry import func
from tilefoundry.dsl import DimVar, ReduceKind, Tensor, tf

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 4
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 8 query heads share one kv head
Q_PROJ = NUM_Q_HEADS * HEAD_DIM  # 4096 == DIM
KV_PROJ = NUM_KV_HEADS * HEAD_DIM  # 512

# One new token decoded per step.
S = 1

# Decode ceiling for tonight's fixture (M3 exercises ctx 512 and 4096 against
# the same built Functions — this only bounds the DimVar envelope).
MAX_CTX = 8192

P = DimVar(name="dsv4_prior_ctx", lo=1, hi=MAX_CTX)
C = DimVar(name="dsv4_ctx_total", lo=1, hi=MAX_CTX + 1)
ROPE_CACHE = DimVar(name="dsv4_rope_cache", lo=1, hi=MAX_CTX + 1)


@func
def attn_kv_update(
    hidden: Tensor[(1, S, DIM), "bf16"],
    gamma_in: Tensor[(DIM,), "bf16"],
    w_k: Tensor[(1, DIM, KV_PROJ), "bf16"],
    w_v: Tensor[(1, DIM, KV_PROJ), "bf16"],
    gamma_k: Tensor[(HEAD_DIM,), "bf16"],
    cos_cache: Tensor[(ROPE_CACHE, HEAD_DIM), "bf16"],
    sin_cache: Tensor[(ROPE_CACHE, HEAD_DIM), "bf16"],
    pos_ids: Tensor[(S,), "i32"],
    k_cache_prev: Tensor[(1, P, NUM_KV_HEADS, HEAD_DIM), "bf16"],
    v_cache_prev: Tensor[(1, P, NUM_KV_HEADS, HEAD_DIM), "bf16"],
):
    # Full cache after key RMSNorm + RoPE + append: (k_full, v_full) of shape
    # [1, prior+1, kv_heads, head_dim] (returned, not matmul'd).
    hidden_norm = tf.rms_norm(hidden, gamma_in)
    k = tf.reshape(tf.matmul(hidden_norm, w_k), new_shape=(1, S, NUM_KV_HEADS, HEAD_DIM))
    v = tf.reshape(tf.matmul(hidden_norm, w_v), new_shape=(1, S, NUM_KV_HEADS, HEAD_DIM))
    k_norm = tf.rms_norm(k, gamma_k)
    _, k_rope = tf.rope(k_norm, k_norm, cos_cache, sin_cache, pos_ids)
    k_full = tf.concat(k_cache_prev, k_rope, axis=1)
    v_full = tf.concat(v_cache_prev, v, axis=1)
    return (k_full, v_full)


@func
def attn_scores(
    hidden: Tensor[(1, S, DIM), "bf16"],
    gamma_in: Tensor[(DIM,), "bf16"],
    w_q: Tensor[(1, DIM, Q_PROJ), "bf16"],
    gamma_q: Tensor[(HEAD_DIM,), "bf16"],
    cos_cache: Tensor[(ROPE_CACHE, HEAD_DIM), "bf16"],
    sin_cache: Tensor[(ROPE_CACHE, HEAD_DIM), "bf16"],
    pos_ids: Tensor[(S,), "i32"],
    k_full: Tensor[(1, C, NUM_KV_HEADS, HEAD_DIM), "bf16"],
    v_full: Tensor[(1, C, NUM_KV_HEADS, HEAD_DIM), "bf16"],
    attn_mask: Tensor[(1, 1, S, C), "bf16"],
    scale: Tensor[(1, 1, 1, 1), "bf16"],
    w_o: Tensor[(1, Q_PROJ, DIM), "bf16"],
) -> Tensor[(1, S, DIM), "bf16"]:
    # GQA attention of the new token's query over the full (dynamic-length)
    # context, then the output projection.
    hidden_norm = tf.rms_norm(hidden, gamma_in)
    q = tf.reshape(tf.matmul(hidden_norm, w_q), new_shape=(1, S, NUM_Q_HEADS, HEAD_DIM))
    q_norm = tf.rms_norm(q, gamma_q)
    q_rope, _ = tf.rope(q_norm, q_norm, cos_cache, sin_cache, pos_ids)

    k_b = tf.repeat_interleave(k_full, repeats=GQA_GROUP, axis=2)
    v_b = tf.repeat_interleave(v_full, repeats=GQA_GROUP, axis=2)
    q_h = tf.transpose(q_rope, perm=(0, 2, 1, 3))
    k_h = tf.transpose(k_b, perm=(0, 2, 1, 3))
    v_h = tf.transpose(v_b, perm=(0, 2, 1, 3))
    q_s = tf.mul(q_h, scale)
    k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
    scores = tf.add(tf.matmul(q_s, k_t), attn_mask)
    probs = tf.softmax(scores, axis=-1)
    ctx = tf.matmul(probs, v_h)
    attn_out = tf.transpose(ctx, perm=(0, 2, 1, 3))
    return tf.matmul(tf.reshape(attn_out, new_shape=(1, S, Q_PROJ)), w_o)


# ═════════════════════════════════════════════════════════════════════════
# Stage 2 (best-effort, new symbols only): the GQA placeholder above is left
# byte-for-byte untouched — decode_step.py / test_decode_step_e2e.py, owned
# by another agent working in this worktree, import attn_kv_update /
# attn_scores / HEAD_DIM / NUM_Q_HEADS / NUM_KV_HEADS / GQA_GROUP /
# Q_PROJ / KV_PROJ / S / P / C / ROPE_CACHE / MAX_CTX directly.
#
# Real-dimension DeepSeek-V4-Flash MLA attention decode (single new token,
# start_pos > 0 -- matching this file's own existing decode-only scope).
# Scope: real transformer layer 0 (config.json ``compress_ratios[0] == 0``)
# -- pure sliding-window MLA: low-rank Q (wq_a -> q_norm -> wq_b), a single
# shared 512-dim KV latent (wkv -> kv_norm; MQA, n_kv_heads == 1), a FIXED
# 128-token sliding-window cache (no unbounded growth -> a genuine static
# shape, no DimVar needed here unlike the placeholder above), attn_sink
# -augmented softmax, grouped low-rank O projection (wo_a per-group ->
# wo_b). Cross-checked against this worktree's stage-1 torch oracle,
# ``tests/models/deepseek_v4_flash/hf_attention_ref.py``'s
# ``AttentionRef(layer_id=0, ...)``, via ``test_attention_hir_oracle.py``.
#
# NOT attempted: real layers with compress_ratio != 0 (the learned Indexer +
# Compressor top-k KV-compression sparsity -- e.g. layer 2, the layer
# hf_attention_ref.py's oracle uses for its harder / index_topk coverage).
# See the run report for the concrete op-contract reasons: TopK's static
# ``k`` attribute vs. index_topk's context-length-dependent
# ``min(512, end_pos // 4)``; Gather's ``batch_dims`` convention not
# matching the per-(batch,query-position)-own-index-set KV gather the
# Indexer's selection needs; the Compressor's conditional/stateful
# ring-buffer update (an ``if should_compress: ... else: return None``
# gating a cache write) not obviously expressible as a pure dataflow
# Function.
#
# ``tilefoundry.dsl.tf.rope`` implements the "rotate_half" (half-split)
# convention (see its evaluator: ``q*cos + rotate_half(q)*sin``);
# DeepSeek-V4's official ``apply_rotary_emb`` (hf_attention_ref.py, ported
# from model.py) uses the "interleaved-pairs" convention (view-as-complex on
# adjacent pairs (x0,x1),(x2,x3),...). These rotate different dimension
# pairings and are not interchangeable without a permutation adapter (not
# attempted here) -- RoPE below is instead built from primitives (strided
# ``tf.slice`` to de-interleave even/odd elements, ``tf.reshape``+
# ``tf.concat`` to re-interleave), reproducing the official interleaved-pairs
# math exactly rather than using the ``tf.rope`` op. Confirmed against
# ``Slice``'s evaluator (builds a literal python ``slice(begin,end,stride)``
# per axis, so a stride-2 slice is exactly ``x[offset::2]``).
# ═════════════════════════════════════════════════════════════════════════

REAL_DIM = 4096
REAL_N_HEADS = 64
REAL_HEAD_DIM = 512
REAL_ROPE_DIM = 64
REAL_ROPE_HALF = REAL_ROPE_DIM // 2  # 32
REAL_Q_LORA_RANK = 1024
REAL_WINDOW = 128
REAL_O_GROUPS = 8
REAL_O_LORA_RANK = 1024
REAL_Q_PROJ = REAL_N_HEADS * REAL_HEAD_DIM                     # 32768
REAL_WO_A_IN = REAL_N_HEADS * REAL_HEAD_DIM // REAL_O_GROUPS   # 4096
REAL_WO_A_OUT = REAL_O_GROUPS * REAL_O_LORA_RANK               # 8192
REAL_NOPE_DIM = REAL_HEAD_DIM - REAL_ROPE_DIM                  # 448

# KV-cache fp8 fake-quant (official: kernel.py's `act_quant(..., inplace=True)`
# / hf_attention_ref._fake_quant_fp8_block, round_scale=True): per-block
# absmax -> power-of-2 scale, applied to the non-rope portion of the cached
# KV latent only (see mla_kv_update_v2 below).
KV_QUANT_BLOCK = 64                                # official FP8_ACT_BLOCK_SIZE
KV_QUANT_BLOCKS = REAL_NOPE_DIM // KV_QUANT_BLOCK  # 7
FP8E4M3_MAX = 448.0            # max finite magnitude representable in e4m3
FP8E4M3_QUANT_EPS = 1e-4       # amax floor (guards log2(0) on an all-zero block)


# NOTE on `@func` bodies: they are parsed from the Python AST at decoration
# time (`tilefoundry.parser.hir_parser`), not eagerly traced -- a call must
# resolve to a registered Op or another `@func`/`Module` reference; a plain
# (undecorated) Python helper function is rejected ("unknown Op name ...").
# So RoPE (which needs the same interleave/de-interleave math at 3 call
# sites) and the 8-group `wo_a` projection are inlined/unrolled below rather
# than factored into a shared helper or a Python `for` loop -- verbose, but
# every statement is a direct, individually-proven `tf.<op>(...)` call (see
# qwen3_5_30b_a3b/qwen3_module.py and this file's own GQA placeholder above
# for precedent on each op's calling convention).
#
# `mla_kv_update_v2` and `mla_attend_v2` are two separate `@func`s, chained
# in Python by the caller (test_attention_hir_oracle.py) via two `evaluate()`
# calls -- mirroring this file's own placeholder's
# attn_kv_update -> attn_scores split (see its docstring) -- rather than one
# `@func` calling the other internally: only *methods on the same `@module`
# class* calling each other is demonstrated elsewhere (qwen3_module.py); a
# standalone `@func` calling another standalone `@func` has no such
# precedent in this codebase, so it is avoided here rather than assumed.


@func
def mla_kv_update_v2(
    hidden: Tensor[(1, 1, REAL_DIM), "bf16"],
    gamma_kv: Tensor[(REAL_HEAD_DIM,), "bf16"],
    w_kv: Tensor[(REAL_DIM, REAL_HEAD_DIM), "bf16"],
    cos_pos: Tensor[(1, 1, 1, REAL_ROPE_HALF), "f32"],
    sin_pos: Tensor[(1, 1, 1, REAL_ROPE_HALF), "f32"],
    kv_cache0: Tensor[(1, REAL_WINDOW, 1, REAL_HEAD_DIM), "bf16"],
    cur_pos: Tensor[(1,), "i32"],
    s: Tensor[(1,), "i32"],
) -> Tensor[(1, REAL_WINDOW, 1, REAL_HEAD_DIM), "bf16"]:
    # Single shared 512-dim KV latent (MQA, n_kv_heads==1): wkv -> kv_norm ->
    # RoPE on the last 64 dims (interleaved-pairs convention, inlined -- see
    # note above and hf_attention_ref.apply_rotary_emb) -> functional
    # fixed-capacity cache write.
    kv = tf.matmul(hidden, w_kv)
    kv_n = tf.rms_norm(kv, gamma_kv)
    kv_4d = tf.reshape(kv_n, new_shape=(1, 1, 1, REAL_HEAD_DIM))
    kv_nope = tf.slice(kv_4d, begin=(0, 0, 0, 0), end=(1, 1, 1, REAL_NOPE_DIM), strides=(1, 1, 1, 1))
    kv_rope_in = tf.slice(kv_4d, begin=(0, 0, 0, REAL_NOPE_DIM), end=(1, 1, 1, REAL_HEAD_DIM), strides=(1, 1, 1, 1))

    # Official additionally fake-quantizes the cached KV latent's non-rope
    # portion through an FP8 e4m3 grid with a power-of-2 ("ue8m0") block
    # scale before caching (QAT-noise simulation; hf_attention_ref.
    # _fake_quant_fp8_block / kernel.py's `act_quant(..., inplace=True)`,
    # round_scale=True): reshape into 64-wide blocks, block-absmax -> clamp
    # to a floor -> round the scale up to a power of 2 (exp2(ceil(log2(.)))
    # -- needs the CEIL/EXP2/LOG2 unary ops) -> divide -> clamp to the fp8
    # range -> real fp8e4m3 cast round-trip -> multiply back by the scale.
    # kv_rope_in (the last REAL_ROPE_DIM dims) is intentionally left
    # bf16/unquantized, matching the official "rope dims kept for
    # positional precision" comment (model.py).
    kv_nope_f32 = tf.cast(kv_nope, dtype="f32")
    kv_nope_blk = tf.reshape(kv_nope_f32, new_shape=(1, 1, 1, KV_QUANT_BLOCKS, KV_QUANT_BLOCK))
    kv_amax = tf.reduce(kv_nope_blk, axes=(-1,), keepdim=True, kind=ReduceKind.ABS_MAX)
    kv_amax = tf.max(kv_amax, FP8E4M3_QUANT_EPS)
    kv_scale = tf.exp2(tf.ceil(tf.log2(tf.div(kv_amax, FP8E4M3_MAX))))
    kv_scaled = tf.div(kv_nope_blk, kv_scale)
    kv_scaled = tf.min(tf.max(kv_scaled, -FP8E4M3_MAX), FP8E4M3_MAX)
    kv_q_fp8 = tf.cast(kv_scaled, dtype="fp8e4m3")
    kv_dq = tf.mul(tf.cast(kv_q_fp8, dtype="f32"), kv_scale)
    kv_nope_q = tf.cast(tf.reshape(kv_dq, new_shape=(1, 1, 1, REAL_NOPE_DIM)), dtype="bf16")

    # f32 upcast for the rotation itself, single rounding back to bf16 at the
    # end -- matches official apply_rotary_emb's x.float() ... y.copy_(x)
    # (see hf_attention_ref.py); cos_pos/sin_pos are f32-typed (see signature)
    # so no separate cast is needed for them.
    kv_r0 = tf.slice(kv_rope_in, begin=(0, 0, 0, 0), end=(1, 1, 1, REAL_ROPE_DIM), strides=(1, 1, 1, 2))
    kv_r1 = tf.slice(kv_rope_in, begin=(0, 0, 0, 1), end=(1, 1, 1, REAL_ROPE_DIM), strides=(1, 1, 1, 2))
    kv_r0_f32 = tf.cast(kv_r0, dtype="f32")
    kv_r1_f32 = tf.cast(kv_r1, dtype="f32")
    kv_o0_f32 = tf.sub(tf.mul(kv_r0_f32, cos_pos), tf.mul(kv_r1_f32, sin_pos))
    kv_o1_f32 = tf.add(tf.mul(kv_r0_f32, sin_pos), tf.mul(kv_r1_f32, cos_pos))
    kv_o0 = tf.cast(kv_o0_f32, dtype="bf16")
    kv_o1 = tf.cast(kv_o1_f32, dtype="bf16")
    kv_o0 = tf.reshape(kv_o0, new_shape=(1, 1, 1, REAL_ROPE_HALF, 1))
    kv_o1 = tf.reshape(kv_o1, new_shape=(1, 1, 1, REAL_ROPE_HALF, 1))
    kv_interleaved = tf.concat(kv_o0, kv_o1, axis=-1)
    kv_rope_out = tf.reshape(kv_interleaved, new_shape=(1, 1, 1, REAL_ROPE_DIM))
    kv_final = tf.concat(kv_nope_q, kv_rope_out, axis=-1)
    return tf.cache_update(kv_cache0, cur_pos, s, kv_final)


@func
def mla_attend_v2(
    hidden: Tensor[(1, 1, REAL_DIM), "bf16"],
    gamma_q_lora: Tensor[(REAL_Q_LORA_RANK,), "bf16"],
    w_q_a: Tensor[(REAL_DIM, REAL_Q_LORA_RANK), "bf16"],
    w_q_b: Tensor[(REAL_Q_LORA_RANK, REAL_Q_PROJ), "bf16"],
    ones_head_dim: Tensor[(REAL_HEAD_DIM,), "bf16"],
    cos_pos: Tensor[(1, 1, 1, REAL_ROPE_HALF), "f32"],
    sin_pos: Tensor[(1, 1, 1, REAL_ROPE_HALF), "f32"],
    kv_cache: Tensor[(1, REAL_WINDOW, 1, REAL_HEAD_DIM), "bf16"],
    attn_mask: Tensor[(1, 1, 1, REAL_WINDOW), "bf16"],
    attn_sink: Tensor[(1, REAL_N_HEADS, 1, 1), "bf16"],
    scale: Tensor[(1, 1, 1, 1), "bf16"],
    w_o_a: Tensor[(REAL_O_GROUPS, REAL_WO_A_IN, REAL_O_LORA_RANK), "bf16"],
    w_o_b: Tensor[(REAL_WO_A_OUT, REAL_DIM), "bf16"],
) -> Tensor[(1, 1, REAL_DIM), "bf16"]:
    # Low-rank Q (wq_a -> q_norm -> wq_b), per-head unweighted RMS rescale
    # (official: ``q *= rsqrt(mean(q**2,-1)+eps)``, no learned weight --
    # reproduced via ``tf.rms_norm`` with an all-ones weight; official does
    # this one step without an fp32 upcast, rms_norm's evaluator upcasts
    # internally like its other calls -- a minor, flagged precision-only
    # deviation, see report), RoPE on the last 64 dims (inlined, see note
    # above this function).
    q_lat = tf.rms_norm(tf.matmul(hidden, w_q_a), gamma_q_lora)
    q_full = tf.matmul(q_lat, w_q_b)
    q = tf.reshape(q_full, new_shape=(1, 1, REAL_N_HEADS, REAL_HEAD_DIM))
    q_rescaled = tf.rms_norm(q, ones_head_dim)
    q_nope = tf.slice(q_rescaled, begin=(0, 0, 0, 0), end=(1, 1, REAL_N_HEADS, REAL_NOPE_DIM), strides=(1, 1, 1, 1))
    q_rope_in = tf.slice(
        q_rescaled, begin=(0, 0, 0, REAL_NOPE_DIM), end=(1, 1, REAL_N_HEADS, REAL_HEAD_DIM), strides=(1, 1, 1, 1),
    )
    # f32 upcast for the rotation itself, single rounding back to bf16 (see
    # mla_kv_update_v2's identical rope block for the rationale).
    q_r0 = tf.slice(q_rope_in, begin=(0, 0, 0, 0), end=(1, 1, REAL_N_HEADS, REAL_ROPE_DIM), strides=(1, 1, 1, 2))
    q_r1 = tf.slice(q_rope_in, begin=(0, 0, 0, 1), end=(1, 1, REAL_N_HEADS, REAL_ROPE_DIM), strides=(1, 1, 1, 2))
    q_r0_f32 = tf.cast(q_r0, dtype="f32")
    q_r1_f32 = tf.cast(q_r1, dtype="f32")
    q_o0_f32 = tf.sub(tf.mul(q_r0_f32, cos_pos), tf.mul(q_r1_f32, sin_pos))
    q_o1_f32 = tf.add(tf.mul(q_r0_f32, sin_pos), tf.mul(q_r1_f32, cos_pos))
    q_o0 = tf.cast(q_o0_f32, dtype="bf16")
    q_o1 = tf.cast(q_o1_f32, dtype="bf16")
    q_o0 = tf.reshape(q_o0, new_shape=(1, 1, REAL_N_HEADS, REAL_ROPE_HALF, 1))
    q_o1 = tf.reshape(q_o1, new_shape=(1, 1, REAL_N_HEADS, REAL_ROPE_HALF, 1))
    q_interleaved = tf.concat(q_o0, q_o1, axis=-1)
    q_rope_out = tf.reshape(q_interleaved, new_shape=(1, 1, REAL_N_HEADS, REAL_ROPE_DIM))
    q_final = tf.concat(q_nope, q_rope_out, axis=-1)

    # MQA broadcast (n_kv_heads==1 -> REAL_N_HEADS via repeat_interleave, same
    # op/pattern as this file's own GQA placeholder above and qwen3_module.py);
    # kv_cache serves as both K and V (MLA-absorbed: no separate V projection).
    k_b = tf.repeat_interleave(kv_cache, repeats=REAL_N_HEADS, axis=2)
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
        probs_ext, begin=(0, 0, 0, 0), end=(1, REAL_N_HEADS, 1, REAL_WINDOW), strides=(1, 1, 1, 1),
    )
    ctx = tf.matmul(probs, k_h)

    # Inverse-RoPE the attention output's last 64 dims (official:
    # ``apply_rotary_emb(o[...,-rd:], freqs_cis, True)``, same query
    # position; inverse uses the conjugate angle: (x0*cos+x1*sin,
    # x1*cos-x0*sin) -- see hf_attention_ref.apply_rotary_emb).
    ctx_nope = tf.slice(ctx, begin=(0, 0, 0, 0), end=(1, REAL_N_HEADS, 1, REAL_NOPE_DIM), strides=(1, 1, 1, 1))
    ctx_rope_in = tf.slice(
        ctx, begin=(0, 0, 0, REAL_NOPE_DIM), end=(1, REAL_N_HEADS, 1, REAL_HEAD_DIM), strides=(1, 1, 1, 1),
    )
    # f32 upcast for the rotation itself, single rounding back to bf16 (see
    # mla_kv_update_v2's identical rope block for the rationale).
    ctx_r0 = tf.slice(ctx_rope_in, begin=(0, 0, 0, 0), end=(1, REAL_N_HEADS, 1, REAL_ROPE_DIM), strides=(1, 1, 1, 2))
    ctx_r1 = tf.slice(ctx_rope_in, begin=(0, 0, 0, 1), end=(1, REAL_N_HEADS, 1, REAL_ROPE_DIM), strides=(1, 1, 1, 2))
    ctx_r0_f32 = tf.cast(ctx_r0, dtype="f32")
    ctx_r1_f32 = tf.cast(ctx_r1, dtype="f32")
    ctx_o0_f32 = tf.add(tf.mul(ctx_r0_f32, cos_pos), tf.mul(ctx_r1_f32, sin_pos))
    ctx_o1_f32 = tf.sub(tf.mul(ctx_r1_f32, cos_pos), tf.mul(ctx_r0_f32, sin_pos))
    ctx_o0 = tf.cast(ctx_o0_f32, dtype="bf16")
    ctx_o1 = tf.cast(ctx_o1_f32, dtype="bf16")
    ctx_o0 = tf.reshape(ctx_o0, new_shape=(1, REAL_N_HEADS, 1, REAL_ROPE_HALF, 1))
    ctx_o1 = tf.reshape(ctx_o1, new_shape=(1, REAL_N_HEADS, 1, REAL_ROPE_HALF, 1))
    ctx_interleaved = tf.concat(ctx_o0, ctx_o1, axis=-1)
    ctx_rope_out = tf.reshape(ctx_interleaved, new_shape=(1, REAL_N_HEADS, 1, REAL_ROPE_DIM))
    ctx_final = tf.concat(ctx_nope, ctx_rope_out, axis=-1)

    attn_out_heads_last = tf.transpose(ctx_final, perm=(0, 2, 1, 3))
    o_flat = tf.reshape(attn_out_heads_last, new_shape=(1, 1, REAL_Q_PROJ))

    # Grouped low-rank O projection (wo_a): official reinterprets one
    # [WO_A_OUT, WO_A_IN] weight as REAL_O_GROUPS(==8) independent
    # [O_LORA_RANK, WO_A_IN] blocks, each applied to its own
    # (8-consecutive-heads) input slice (``torch.einsum("bsgd,grd->bsgr",
    # o, wo_a)`` in model.py). Unrolled below into 8 explicit
    # slice+reshape+matmul groups (a Python `for` loop building a list is
    # not used -- see note above this function) rather than relying on an
    # unverified N-D-broadcasting batched matmul; group g covers
    # o_flat[:, :, g*WO_A_IN : (g+1)*WO_A_IN] against w_o_a[g].
    o_g0 = tf.slice(o_flat, begin=(0, 0, 0), end=(1, 1, 4096), strides=(1, 1, 1))
    o_g1 = tf.slice(o_flat, begin=(0, 0, 4096), end=(1, 1, 8192), strides=(1, 1, 1))
    o_g2 = tf.slice(o_flat, begin=(0, 0, 8192), end=(1, 1, 12288), strides=(1, 1, 1))
    o_g3 = tf.slice(o_flat, begin=(0, 0, 12288), end=(1, 1, 16384), strides=(1, 1, 1))
    o_g4 = tf.slice(o_flat, begin=(0, 0, 16384), end=(1, 1, 20480), strides=(1, 1, 1))
    o_g5 = tf.slice(o_flat, begin=(0, 0, 20480), end=(1, 1, 24576), strides=(1, 1, 1))
    o_g6 = tf.slice(o_flat, begin=(0, 0, 24576), end=(1, 1, 28672), strides=(1, 1, 1))
    o_g7 = tf.slice(o_flat, begin=(0, 0, 28672), end=(1, 1, 32768), strides=(1, 1, 1))

    w_g0 = tf.reshape(tf.slice(w_o_a, begin=(0, 0, 0), end=(1, 4096, 1024), strides=(1, 1, 1)), new_shape=(4096, 1024))
    w_g1 = tf.reshape(tf.slice(w_o_a, begin=(1, 0, 0), end=(2, 4096, 1024), strides=(1, 1, 1)), new_shape=(4096, 1024))
    w_g2 = tf.reshape(tf.slice(w_o_a, begin=(2, 0, 0), end=(3, 4096, 1024), strides=(1, 1, 1)), new_shape=(4096, 1024))
    w_g3 = tf.reshape(tf.slice(w_o_a, begin=(3, 0, 0), end=(4, 4096, 1024), strides=(1, 1, 1)), new_shape=(4096, 1024))
    w_g4 = tf.reshape(tf.slice(w_o_a, begin=(4, 0, 0), end=(5, 4096, 1024), strides=(1, 1, 1)), new_shape=(4096, 1024))
    w_g5 = tf.reshape(tf.slice(w_o_a, begin=(5, 0, 0), end=(6, 4096, 1024), strides=(1, 1, 1)), new_shape=(4096, 1024))
    w_g6 = tf.reshape(tf.slice(w_o_a, begin=(6, 0, 0), end=(7, 4096, 1024), strides=(1, 1, 1)), new_shape=(4096, 1024))
    w_g7 = tf.reshape(tf.slice(w_o_a, begin=(7, 0, 0), end=(8, 4096, 1024), strides=(1, 1, 1)), new_shape=(4096, 1024))

    y0 = tf.matmul(o_g0, w_g0)
    y1 = tf.matmul(o_g1, w_g1)
    y2 = tf.matmul(o_g2, w_g2)
    y3 = tf.matmul(o_g3, w_g3)
    y4 = tf.matmul(o_g4, w_g4)
    y5 = tf.matmul(o_g5, w_g5)
    y6 = tf.matmul(o_g6, w_g6)
    y7 = tf.matmul(o_g7, w_g7)

    y = tf.concat(y0, y1, axis=-1)
    y = tf.concat(y, y2, axis=-1)
    y = tf.concat(y, y3, axis=-1)
    y = tf.concat(y, y4, axis=-1)
    y = tf.concat(y, y5, axis=-1)
    y = tf.concat(y, y6, axis=-1)
    y = tf.concat(y, y7, axis=-1)
    return tf.matmul(y, w_o_b)


__all__ = [
    "C",
    "GQA_GROUP",
    "HEAD_DIM",
    "KV_PROJ",
    "MAX_CTX",
    "NUM_KV_HEADS",
    "NUM_Q_HEADS",
    "P",
    "Q_PROJ",
    "ROPE_CACHE",
    "S",
    "attn_kv_update",
    "attn_scores",
    "REAL_DIM",
    "REAL_HEAD_DIM",
    "REAL_N_HEADS",
    "REAL_NOPE_DIM",
    "REAL_O_GROUPS",
    "REAL_O_LORA_RANK",
    "REAL_Q_LORA_RANK",
    "REAL_Q_PROJ",
    "REAL_ROPE_DIM",
    "REAL_ROPE_HALF",
    "REAL_WINDOW",
    "REAL_WO_A_IN",
    "REAL_WO_A_OUT",
    "mla_attend_v2",
    "mla_kv_update_v2",
]
