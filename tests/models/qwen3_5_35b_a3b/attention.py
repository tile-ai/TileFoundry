"""Qwen3.5-35B-A3B full-attention component: 10 of the 40 decoder layers
(``layer_types`` selects full_attention vs. linear_attention per layer; the
other 30 use ``gdn.py``'s GatedDeltaNet). Mirrors
``transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`` (transformers
5.12.1, cited below as M).

``full_attn_mix``: input RMSNorm -> q_proj emits per-head [query(256)|
gate(256)] (M:684-688) -> per-head q/k RMSNorm -> partial RoPE on the first
``ROT_DIM`` dims, rotate-half convention (M:568, same as ``tf.rope``) ->
KV-cache write -> GQA 8:1 attend with an additive causal mask -> sigmoid
output gate (M:716-717) -> o_proj -> residual. The causal mask follows the
qwen3_5_30b_a3b fixture convention: the caller supplies an additive mask (0
on the valid prefix, -inf elsewhere).

``attn_convert``: the sole RAW-checkpoint -> CANONICAL (kernel-native
layout) weight conversion entry point for ``full_attn_mix`` -- pure
repack/cast, no numeric change. RMSNorm weights are checkpoint-stored as
``w - 1`` (M:817); ``attn_convert`` keeps norms RAW and ``full_attn_mix``
applies ``(1 + w)`` in-body.
"""
from __future__ import annotations

from tests.models.qwen3_5_35b_a3b.config import HIDDEN
from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.ir.core.kinds import ReduceKind
from tilefoundry.ir.core.module import Module

N_Q_HEADS = 16
N_KV_HEADS = 2
HEAD_DIM = 256
ROT_DIM = 64          # partial_rotary_factor 0.25 x 256
Q_PROJ = N_Q_HEADS * HEAD_DIM              # 4096
QG_PROJ = Q_PROJ * 2                       # 8192: per-head [query(256) | gate(256)]
KV_PROJ = N_KV_HEADS * HEAD_DIM            # 512
GQA_GROUP = N_Q_HEADS // N_KV_HEADS        # 8
SCALE = HEAD_DIM ** -0.5

CACHE_CAP = 4096


@func
def full_attn_mix(
    x: Tensor[(1, 1, HIDDEN), "bf16"],                       # layer residual stream
    in_norm_raw: ConstTensor[(HIDDEN,), "f32"],              # RAW (not +1); (1+w) applied in-body
    w_qg: ConstTensor[(HIDDEN, QG_PROJ), "bf16"],
    w_k: ConstTensor[(HIDDEN, KV_PROJ), "bf16"],
    w_v: ConstTensor[(HIDDEN, KV_PROJ), "bf16"],
    w_o: ConstTensor[(Q_PROJ, HIDDEN), "bf16"],
    q_norm_raw: ConstTensor[(HEAD_DIM,), "f32"],             # RAW (not +1)
    k_norm_raw: ConstTensor[(HEAD_DIM,), "f32"],
    cos_cache: Tensor[(CACHE_CAP, ROT_DIM), "f32"],          # rope table (full length, indexed by pos_ids)
    sin_cache: Tensor[(CACHE_CAP, ROT_DIM), "f32"],
    pos_ids: Tensor[(1, 1), "i32"],                          # current position id
    k_cache: Tensor[(1, CACHE_CAP, N_KV_HEADS, HEAD_DIM), "bf16"],
    v_cache: Tensor[(1, CACHE_CAP, N_KV_HEADS, HEAD_DIM), "bf16"],
    pos: Tensor[(1,), "i32"],                                # write slot = current length
    s_one: Tensor[(1,), "i32"],                              # constant 1 (cache_update's s)
    attn_mask: Tensor[(1, 1, 1, CACHE_CAP), "f32"],          # additive: 0 up to pos, -inf beyond
):
    # Returns (y[1,1,2048], k_cache', v_cache') -- multi-output funcs carry no
    # return annotation, per this repo's convention (inferred).
    # CANONICAL weight layout: q/k/v/o already transposed [in,out], the 3
    # norms stay RAW -- evaluator (this func) and the kernel override read
    # the same self.weights (sole conversion entry: attn_convert).
    # ── input_layernorm (M:855, RAW weight, (1+w) applied in-body) ──────────
    h = tf.rms_norm(x, 1.0 + in_norm_raw)

    # ── q_proj -> per-head [query|gate] (M:684-688) ──────────────────────────
    qg = tf.reshape(tf.matmul(h, w_qg), new_shape=(1, 1, N_Q_HEADS, 2 * HEAD_DIM))
    q = tf.slice(qg, begin=(0, 0, 0, 0), end=(1, 1, N_Q_HEADS, HEAD_DIM), strides=(1, 1, 1, 1))
    gate = tf.slice(qg, begin=(0, 0, 0, HEAD_DIM), end=(1, 1, N_Q_HEADS, 2 * HEAD_DIM), strides=(1, 1, 1, 1))
    k = tf.reshape(tf.matmul(h, w_k), new_shape=(1, 1, N_KV_HEADS, HEAD_DIM))
    v = tf.reshape(tf.matmul(h, w_v), new_shape=(1, 1, N_KV_HEADS, HEAD_DIM))

    # ── per-head RMSNorm (M:690-692, RAW weight, (1+w) applied in-body) ──────
    q = tf.rms_norm(q, 1.0 + q_norm_raw)
    k = tf.rms_norm(k, 1.0 + k_norm_raw)

    # ── partial rope: first ROT_DIM dims, rotate-half convention (M:568) ─────
    q_rot = tf.slice(q, begin=(0, 0, 0, 0), end=(1, 1, N_Q_HEADS, ROT_DIM), strides=(1, 1, 1, 1))
    q_pass = tf.slice(q, begin=(0, 0, 0, ROT_DIM), end=(1, 1, N_Q_HEADS, HEAD_DIM), strides=(1, 1, 1, 1))
    k_rot = tf.slice(k, begin=(0, 0, 0, 0), end=(1, 1, N_KV_HEADS, ROT_DIM), strides=(1, 1, 1, 1))
    k_pass = tf.slice(k, begin=(0, 0, 0, ROT_DIM), end=(1, 1, N_KV_HEADS, HEAD_DIM), strides=(1, 1, 1, 1))
    q_rot_r, k_rot_r = tf.rope(
        tf.cast(q_rot, dtype="f32"), tf.cast(k_rot, dtype="f32"), cos_cache, sin_cache, pos_ids
    )
    q = tf.concat(tf.cast(q_rot_r, dtype="bf16"), q_pass, axis=-1)
    k = tf.concat(tf.cast(k_rot_r, dtype="bf16"), k_pass, axis=-1)

    # ── write cache at current position (cache_update: new[:, :s] -> cache[:, pos:pos+s]) ─
    k_cache_new = tf.cache_update(k_cache, pos, s_one, k)
    v_cache_new = tf.cache_update(v_cache, pos, s_one, v)

    # ── attend: GQA 8:1, additive mask (M:618 eager; scale folded into q) ────
    qh = tf.reshape(q, new_shape=(1, N_Q_HEADS, 1, HEAD_DIM))
    qh = tf.cast(qh, dtype="f32") * tf.full_like(tf.cast(qh, dtype="f32"), value=SCALE)
    kh = tf.transpose(k_cache_new, perm=(0, 2, 1, 3))         # (1, 2, CAP, 256)
    vh = tf.transpose(v_cache_new, perm=(0, 2, 1, 3))
    kh = tf.repeat_interleave(kh, repeats=GQA_GROUP, axis=1)  # (1, 16, CAP, 256)
    vh = tf.repeat_interleave(vh, repeats=GQA_GROUP, axis=1)
    scores = tf.matmul(qh, tf.transpose(tf.cast(kh, dtype="f32"), perm=(0, 1, 3, 2)))  # (1,16,1,CAP) f32
    scores = scores + attn_mask
    smax = tf.reduce(scores, axes=(-1,), keepdim=True, kind=ReduceKind.MAX)
    p = tf.exp(scores - smax)
    p = p / tf.reduce(p, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
    o = tf.matmul(tf.cast(p, dtype="bf16"), vh)               # (1,16,1,256)

    # ── sigmoid output gate (M:716-717) -> o_proj -> residual ────────────────
    o = o * tf.sigmoid(tf.reshape(gate, new_shape=(1, N_Q_HEADS, 1, HEAD_DIM)))
    o_flat = tf.reshape(tf.transpose(o, perm=(0, 2, 1, 3)), new_shape=(1, 1, Q_PROJ))
    y = tf.matmul(o_flat, w_o)
    return x + y, k_cache_new, v_cache_new


@func
def attn_convert(
    input_layernorm: ConstTensor[(HIDDEN,), "f32"],          # RAW ckpt layernorm weight (not +1)
    q_proj: ConstTensor[(QG_PROJ, HIDDEN), "bf16"],           # nn.Linear-native [out,in]
    k_proj: ConstTensor[(KV_PROJ, HIDDEN), "bf16"],
    v_proj: ConstTensor[(KV_PROJ, HIDDEN), "bf16"],
    o_proj: ConstTensor[(HIDDEN, Q_PROJ), "bf16"],            # nn.Linear-native [out,in]
    q_norm: ConstTensor[(HEAD_DIM,), "f32"],                  # RAW ckpt q_norm weight (not +1)
    k_norm: ConstTensor[(HEAD_DIM,), "f32"],
):
    # Returns (in_norm_raw, w_qg, w_k, w_v, w_o, q_norm_raw, k_norm_raw): the
    # CANONICAL weights for full_attn_mix. Sole conversion entry point;
    # evaluator (via full_attn_mix) and the kernel override read this same
    # product. q/k/v/o transposed to [in,out]; norms stay RAW, (1+w) is
    # applied inside full_attn_mix. Pure repack/cast, no numeric change.
    in_norm_raw = tf.cast(input_layernorm, dtype="f32")
    w_qg = tf.cast(tf.transpose(q_proj, perm=(1, 0)), dtype="bf16")
    w_k = tf.cast(tf.transpose(k_proj, perm=(1, 0)), dtype="bf16")
    w_v = tf.cast(tf.transpose(v_proj, perm=(1, 0)), dtype="bf16")
    w_o = tf.cast(tf.transpose(o_proj, perm=(1, 0)), dtype="bf16")
    q_norm_raw = tf.cast(q_norm, dtype="f32")
    k_norm_raw = tf.cast(k_norm, dtype="f32")
    return in_norm_raw, w_qg, w_k, w_v, w_o, q_norm_raw, k_norm_raw


attention_module = Module(name="attention", functions=(full_attn_mix, attn_convert), entry="full_attn_mix")
