"""Qwen3-1.7B dense decoder layer described as a single tilefoundry IR Module.

Phase 0 companion to ``tests/models/qwen3_5_30b_a3b/qwen3_module.py``: same
``@module class`` authoring style (each kernel is a named ``@func`` method;
the decorator returns the ``tilefoundry.ir.core.module.Module`` that the class
name binds directly to — ``Qwen3_1_7B.self_attention`` pulls a kernel by
attribute, mirroring the HF model). Two things make this contract simpler
than the MoE-30B one:

- the MLP is a single dense SwiGLU expert (plain gate/up/down projection), not
  the 30B sibling's runtime top-k expert routing (no router, no ``topk``, no
  ``gather``);
- there is no KV cache: this is a single-shot prefill oracle over a static
  ``S_CAP``-token sequence (``cur_pos`` is always 0), so every kernel takes
  and returns one plain tensor — no ``cache_update``, no dynamic ``DimVar``,
  and no need to split attention across two ``@func``s to dodge a
  compound-``DimVar`` ``concat`` (the reason ``qwen3_5_30b_a3b/common.py``'s
  attention is ``build_kv_update`` + ``build_scores`` rather than one
  Function: there, the KV cache's ``prior + new`` axis can't feed ``matmul``
  once it's a ``DimVar`` sum; here there is no cache to concat, so
  ``self_attention`` is one ``@func``).

``self_attention`` and ``mlp`` each fuse their preceding RMSNorm internally
(``input_rms_norm`` / the post-attention norm) — matching the Qwen3-30B-A3B
sibling's convention (its ``self_attention`` fuses ``input_rms_norm``; its
``moe`` fuses the post-attention norm) so each fused kernel lines up with one
HF pre-norm-then-block composition. ``decoder_layer`` composes
``self_attention`` + residual + ``mlp`` + residual, mirroring
``Qwen3DecoderLayer.forward`` exactly.

Qwen3's per-head ``q_norm`` / ``k_norm`` (RMSNorm over just the ``head_dim``
axis, applied to every head independently) needs no special HIR combinator:
``tf.rms_norm`` normalizes only the last axis and is rank-agnostic on every
axis before it (see ``tilefoundry/ir/hir/nn/rms_norm.py``), so calling it
directly on the ``[1, S_CAP, heads, head_dim]`` tensor — the same reshape the
head split already produces — reproduces HF's
``q_norm(q_proj(x).view(hidden_shape))`` with no extra reshape either side of
the norm (the Qwen3 HF docstring notes exactly this: "unlike olmo, only on
the head dim... thus post q_norm does not need reshape").
"""
from __future__ import annotations

from tests.models.qwen3_1_7b.common import (
    DT,
    GQA_GROUP,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_PROJ,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    Q_PROJ,
    S_CAP,
)
from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies


@module(entry="decoder_layer")
class Qwen3_1_7B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, S_CAP, HIDDEN), DT],
        gamma_in: Tensor[(HIDDEN,), DT],
    ) -> Tensor[(1, S_CAP, HIDDEN), DT]:
        # Pre-attention input RMSNorm; HF `Qwen3DecoderLayer.input_layernorm`.
        return tf.rms_norm(hidden, gamma_in)

    @func
    def self_attention(
        hidden: Tensor[(1, S_CAP, HIDDEN), DT],
        gamma_in: Tensor[(HIDDEN,), DT],
        w_q: Tensor[(1, HIDDEN, Q_PROJ), DT],
        w_k: Tensor[(1, HIDDEN, KV_PROJ), DT],
        w_v: Tensor[(1, HIDDEN, KV_PROJ), DT],
        gamma_q: Tensor[(HEAD_DIM,), DT],
        gamma_k: Tensor[(HEAD_DIM,), DT],
        cos_cache: Tensor[(S_CAP, HEAD_DIM), DT],
        sin_cache: Tensor[(S_CAP, HEAD_DIM), DT],
        pos_ids: Tensor[(S_CAP,), "i32"],
        attn_mask: Tensor[(1, 1, S_CAP, S_CAP), DT],
        scale: Tensor[(1, 1, 1, 1), DT],
        w_o: Tensor[(1, Q_PROJ, HIDDEN), DT],
    ) -> Tensor[(1, S_CAP, HIDDEN), DT]:
        # Fused input_layernorm + self_attn: GQA + RoPE + per-head q_norm/k_norm,
        # no residual (the layer owns the residual add). Single-shot prefill —
        # no KV cache — so the mask is plain [S_CAP, S_CAP] causal (cur_pos==0).
        hidden_norm = input_rms_norm(hidden, gamma_in)
        q = tf.reshape(tf.matmul(hidden_norm, w_q), new_shape=(1, S_CAP, NUM_Q_HEADS, HEAD_DIM))
        k = tf.reshape(tf.matmul(hidden_norm, w_k), new_shape=(1, S_CAP, NUM_KV_HEADS, HEAD_DIM))
        v = tf.reshape(tf.matmul(hidden_norm, w_v), new_shape=(1, S_CAP, NUM_KV_HEADS, HEAD_DIM))
        q_n = tf.rms_norm(q, gamma_q)
        k_n = tf.rms_norm(k, gamma_k)
        q_rope, _ = tf.rope(q_n, q_n, cos_cache, sin_cache, pos_ids)
        _, k_rope = tf.rope(k_n, k_n, cos_cache, sin_cache, pos_ids)

        k_b = tf.repeat_interleave(k_rope, repeats=GQA_GROUP, axis=2)
        v_b = tf.repeat_interleave(v, repeats=GQA_GROUP, axis=2)
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
    def mlp(
        hidden: Tensor[(1, S_CAP, HIDDEN), DT],
        gamma_post: Tensor[(HIDDEN,), DT],
        w_gate: Tensor[(1, HIDDEN, INTERMEDIATE), DT],
        w_up: Tensor[(1, HIDDEN, INTERMEDIATE), DT],
        w_down: Tensor[(1, INTERMEDIATE, HIDDEN), DT],
    ) -> Tensor[(1, S_CAP, HIDDEN), DT]:
        # Fused post_attention_layernorm + dense SwiGLU, no residual. silu(x) =
        # x * sigmoid(x) — there is no standalone silu op in the HIR op surface.
        hidden_norm = tf.rms_norm(hidden, gamma_post)
        gate = tf.matmul(hidden_norm, w_gate)
        up = tf.matmul(hidden_norm, w_up)
        act = tf.mul(gate, tf.sigmoid(gate))
        h = tf.mul(act, up)
        return tf.matmul(h, w_down)

    @func
    def decoder_layer(
        hidden: Tensor[(1, S_CAP, HIDDEN), DT],
        gamma_in: Tensor[(HIDDEN,), DT],
        w_q: Tensor[(1, HIDDEN, Q_PROJ), DT],
        w_k: Tensor[(1, HIDDEN, KV_PROJ), DT],
        w_v: Tensor[(1, HIDDEN, KV_PROJ), DT],
        gamma_q: Tensor[(HEAD_DIM,), DT],
        gamma_k: Tensor[(HEAD_DIM,), DT],
        cos_cache: Tensor[(S_CAP, HEAD_DIM), DT],
        sin_cache: Tensor[(S_CAP, HEAD_DIM), DT],
        pos_ids: Tensor[(S_CAP,), "i32"],
        attn_mask: Tensor[(1, 1, S_CAP, S_CAP), DT],
        scale: Tensor[(1, 1, 1, 1), DT],
        w_o: Tensor[(1, Q_PROJ, HIDDEN), DT],
        gamma_post: Tensor[(HIDDEN,), DT],
        w_gate: Tensor[(1, HIDDEN, INTERMEDIATE), DT],
        w_up: Tensor[(1, HIDDEN, INTERMEDIATE), DT],
        w_down: Tensor[(1, INTERMEDIATE, HIDDEN), DT],
    ) -> Tensor[(1, S_CAP, HIDDEN), DT]:
        # Full decoder layer: self_attention + residual, then mlp + residual —
        # mirrors `Qwen3DecoderLayer.forward` exactly.
        attn_out = self_attention(
            hidden, gamma_in, w_q, w_k, w_v, gamma_q, gamma_k,
            cos_cache, sin_cache, pos_ids, attn_mask, scale, w_o,
        )
        h1 = tf.add(hidden, attn_out)
        mlp_out = mlp(h1, gamma_post, w_gate, w_up, w_down)
        return tf.add(h1, mlp_out)
