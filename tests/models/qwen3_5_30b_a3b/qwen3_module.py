"""Qwen3-30B-A3B decoder described as a single tilefoundry IR ``Module``.

``@module class Qwen3_30B_A3B`` declares the decoder: each kernel from the tilert
dataflow (``docs/qwen3_30b_a3b/plans/v1/dataflow.drawio``) is a named ``@func``
method, and the decorator returns a ``tilefoundry.ir.core.module.Module`` — the
``Qwen3_30B_A3B`` name binds directly to that Module. Tests pull a single kernel
by attribute (``Qwen3_30B_A3B.self_attention``) and evaluate it against the
corresponding Hugging Face layer — the module mirrors the HF model: ask it for a
layer's function and evaluate it.

The model is bf16 (per-op f32 numerics are covered by the op tests in
``tests/ops``); the kernels declare literal dtypes, matching the module-level
``@func`` convention in ``tests/models/qwen3/test_full_attn_block.py``.

The attention path is modular: ``input_rms_norm``, ``qkv_rope`` (packed QKV
projection + per-head norm + RoPE + KV-cache write), and ``gqa_attend`` (masked
GQA attention + output projection) are named kernels, and ``self_attention``
composes them. The MoE path and the full ``decoder_layer`` / ``decode_step``
land in the same module as they are ported.
"""
from __future__ import annotations

from tests.models.qwen3_5_30b_a3b.common import (
    CACHE_CAP,
    GQA_GROUP,
    HEAD_DIM,
    HIDDEN,
    KV_PROJ,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    Q_PROJ,
    S_CAP,
)
from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies

# Packed q/k/v projection fan-out: one GEMM produces ``[Q_PROJ | KV_PROJ |
# KV_PROJ]`` (4096 + 512 + 512 = 5120), sliced into q/k/v below.
QKV_FAN = Q_PROJ + 2 * KV_PROJ


@module(entry="self_attention")
class Qwen3_30B_A3B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, S_CAP, HIDDEN), "bf16"],
        gamma_in: Tensor[(HIDDEN,), "bf16"],
    ) -> Tensor[(1, S_CAP, HIDDEN), "bf16"]:
        # K1: pre-attention input RMSNorm, kept as its own kernel (feeds qkv_rope).
        return tf.rms_norm(hidden, gamma_in)

    @func
    def qkv_rope(
        hidden_norm: Tensor[(1, S_CAP, HIDDEN), "bf16"],
        w_qkv: Tensor[(1, HIDDEN, QKV_FAN), "bf16"],
        gamma_q: Tensor[(HEAD_DIM,), "bf16"],
        gamma_k: Tensor[(HEAD_DIM,), "bf16"],
        cos_cache: Tensor[(CACHE_CAP, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(CACHE_CAP, HEAD_DIM), "bf16"],
        pos_ids: Tensor[(S_CAP,), "i32"],
        k_cache0: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), "bf16"],
        v_cache0: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        s: Tensor[(1,), "i32"],
    ):
        # K2+K3+K4 fused QkvRope: one packed GEMM ``hidden_norm @ w_qkv`` produces
        # ``[Q_PROJ | KV_PROJ | KV_PROJ]``; slice the q/k/v ranges, per-head
        # RMSNorm q & k, RoPE, then the functional KV cache write at ``cur_pos``.
        qkv = tf.matmul(hidden_norm, w_qkv)
        q_flat = tf.slice(qkv, begin=(0, 0, 0), end=(1, S_CAP, Q_PROJ), strides=(1, 1, 1))
        k_flat = tf.slice(
            qkv, begin=(0, 0, Q_PROJ), end=(1, S_CAP, Q_PROJ + KV_PROJ), strides=(1, 1, 1)
        )
        v_flat = tf.slice(
            qkv, begin=(0, 0, Q_PROJ + KV_PROJ), end=(1, S_CAP, QKV_FAN), strides=(1, 1, 1)
        )
        q = tf.reshape(q_flat, new_shape=(1, S_CAP, NUM_Q_HEADS, HEAD_DIM))
        k = tf.reshape(k_flat, new_shape=(1, S_CAP, NUM_KV_HEADS, HEAD_DIM))
        v = tf.reshape(v_flat, new_shape=(1, S_CAP, NUM_KV_HEADS, HEAD_DIM))
        q_n = tf.rms_norm(q, gamma_q)
        k_n = tf.rms_norm(k, gamma_k)
        q_rope, _ = tf.rope(q_n, q_n, cos_cache, sin_cache, pos_ids)
        _, k_rope = tf.rope(k_n, k_n, cos_cache, sin_cache, pos_ids)
        k_cache1 = tf.cache_update(k_cache0, cur_pos, s, k_rope)
        v_cache1 = tf.cache_update(v_cache0, cur_pos, s, v)
        return (q_rope, k_cache1, v_cache1)

    @func
    def gqa_attend(
        q_rope: Tensor[(1, S_CAP, NUM_Q_HEADS, HEAD_DIM), "bf16"],
        k_cache: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), "bf16"],
        v_cache: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), "bf16"],
        attn_mask: Tensor[(1, 1, S_CAP, CACHE_CAP), "bf16"],
        scale: Tensor[(1, 1, 1, 1), "bf16"],
        w_o: Tensor[(1, Q_PROJ, HIDDEN), "bf16"],
    ) -> Tensor[(1, S_CAP, HIDDEN), "bf16"]:
        # K5+K6: masked GQA attention over the full KV cache, then output
        # projection. KV heads are broadcast to query heads; scores =
        # (q*scale) @ kᵀ + mask.
        k_b = tf.repeat_interleave(k_cache, repeats=GQA_GROUP, axis=2)
        v_b = tf.repeat_interleave(v_cache, repeats=GQA_GROUP, axis=2)
        q_h = tf.transpose(q_rope, perm=(0, 2, 1, 3))
        k_h = tf.transpose(k_b, perm=(0, 2, 1, 3))
        v_h = tf.transpose(v_b, perm=(0, 2, 1, 3))
        q_s = tf.mul(q_h, scale)
        k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
        scores = tf.add(tf.matmul(q_s, k_t), attn_mask)
        probs = tf.softmax(scores, axis=-1)
        ctx = tf.matmul(probs, v_h)
        attn_out = tf.transpose(ctx, perm=(0, 2, 1, 3))
        return tf.matmul(tf.reshape(attn_out, new_shape=(1, S_CAP, Q_PROJ)), w_o)

    @func
    def self_attention(
        hidden: Tensor[(1, S_CAP, HIDDEN), "bf16"],
        gamma_in: Tensor[(HIDDEN,), "bf16"],
        w_qkv: Tensor[(1, HIDDEN, QKV_FAN), "bf16"],
        gamma_q: Tensor[(HEAD_DIM,), "bf16"],
        gamma_k: Tensor[(HEAD_DIM,), "bf16"],
        cos_cache: Tensor[(CACHE_CAP, HEAD_DIM), "bf16"],
        sin_cache: Tensor[(CACHE_CAP, HEAD_DIM), "bf16"],
        pos_ids: Tensor[(S_CAP,), "i32"],
        k_cache0: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), "bf16"],
        v_cache0: Tensor[(1, CACHE_CAP, NUM_KV_HEADS, HEAD_DIM), "bf16"],
        cur_pos: Tensor[(1,), "i32"],
        s: Tensor[(1,), "i32"],
        attn_mask: Tensor[(1, 1, S_CAP, CACHE_CAP), "bf16"],
        scale: Tensor[(1, 1, 1, 1), "bf16"],
        w_o: Tensor[(1, Q_PROJ, HIDDEN), "bf16"],
    ):
        # Decode-step self-attention composed from the named kernels: input
        # RMSNorm, the packed QkvRope (which writes the KV cache at cur_pos),
        # then masked GQA attention + output projection. Returns the output and
        # the updated caches.
        hidden_norm = input_rms_norm(hidden, gamma_in)
        q_rope, k_cache1, v_cache1 = qkv_rope(
            hidden_norm, w_qkv, gamma_q, gamma_k, cos_cache, sin_cache, pos_ids,
            k_cache0, v_cache0, cur_pos, s,
        )
        out = gqa_attend(q_rope, k_cache1, v_cache1, attn_mask, scale, w_o)
        return (out, k_cache1, v_cache1)
