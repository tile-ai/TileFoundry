"""Qwen2.5-1.5B dense decoder layer described as a single tilefoundry IR Module.

Phase 0 companion to ``tests/models/qwen3_1_7b/model/decoder_layer.py``: same
``@module class`` authoring style (each kernel is a named ``@func`` method;
the decorator returns the ``tilefoundry.ir.core.module.Module`` that the class
name binds directly to — ``Qwen2_5_1_5B.lookup("self_attention")`` pulls a kernel by
attribute, mirroring the HF model). As with the Qwen3-1.7B sibling:

- the MLP is a single dense SwiGLU expert (plain gate/up/down projection, no
  router, no ``topk``, no ``gather``);
- there is no KV cache: this is a single-shot prefill oracle over a static
  ``config.s_cap``-token sequence (``cur_pos`` is always 0), so every kernel takes
  and returns one plain tensor and ``self_attention`` is one ``@func`` (no
  ``cache_update`` / two-``@func`` split needed — see the Qwen3-1.7B sibling
  module docstring for why that split exists on the MoE-30B model).

``self_attention`` and ``mlp`` each fuse their preceding RMSNorm internally
(``input_rms_norm`` / the post-attention norm), matching the Qwen3-1.7B
sibling's convention so each fused kernel lines up with one HF
pre-norm-then-block composition. ``decoder_layer`` composes
``self_attention`` + residual + ``mlp`` + residual, mirroring
``Qwen2DecoderLayer.forward`` exactly.

Two differences from the ``qwen3_1_7b`` sibling's ``self_attention``, both
driven by HF ``Qwen2Attention`` (see ``../config.py`` module docstring):

- **no q_norm / k_norm**: Qwen3 applies a per-head RMSNorm to ``q`` and ``k``
  right after the head-split reshape, before RoPE. Qwen2 has no such step —
  RoPE is applied directly to the raw ``q_proj`` / ``k_proj`` output.
- **QKV projection bias**: ``q_proj`` / ``k_proj`` / ``v_proj`` each carry a
  bias (``o_proj`` does not). HF computes ``matmul`` then ``+ bias`` before
  the head-split reshape (``self.q_proj(hidden_states).view(hidden_shape)``,
  and ``nn.Linear`` adds its bias inside that call) — so this HIR mirrors
  that order: ``tf.matmul`` then ``tf.add`` with the bias, then
  ``tf.reshape`` into heads. The bias reshape to ``(1, 1, dim)`` makes the
  broadcast explicit (matching the ``deepseek_v4_flash/moe.py`` gate-bias
  convention) though the HIR ``Binary`` op broadcasts a rank-1 operand
  right-aligned regardless.
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies


@module(entry="decoder_layer")
class Qwen2_5_1_5B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Pre-attention input RMSNorm; HF `Qwen2DecoderLayer.input_layernorm`.
        return tf.rms_norm(hidden, gamma_in)

    @func
    def self_attention(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        bias_q: Tensor[(config.q_proj,), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        bias_k: Tensor[(config.kv_proj,), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        bias_v: Tensor[(config.kv_proj,), config.dt],
        cos_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        sin_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        pos_ids: Tensor[(config.s_cap,), "i32"],
        attn_mask: Tensor[(1, 1, config.s_cap, config.s_cap), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Fused input_layernorm + self_attn: GQA + RoPE, QKV bias, no q_norm/
        # k_norm, no residual (the layer owns the residual add). Single-shot
        # prefill — no KV cache — so the mask is plain [config.s_cap, config.s_cap] causal
        # (cur_pos==0).
        hidden_norm = input_rms_norm(hidden, gamma_in)
        q_lin = tf.add(tf.matmul(hidden_norm, w_q), tf.reshape(bias_q, new_shape=(1, 1, config.q_proj)))
        k_lin = tf.add(tf.matmul(hidden_norm, w_k), tf.reshape(bias_k, new_shape=(1, 1, config.kv_proj)))
        v_lin = tf.add(tf.matmul(hidden_norm, w_v), tf.reshape(bias_v, new_shape=(1, 1, config.kv_proj)))
        q = tf.reshape(q_lin, new_shape=(1, config.s_cap, config.n_q_heads, config.head_dim))
        k = tf.reshape(k_lin, new_shape=(1, config.s_cap, config.n_kv_heads, config.head_dim))
        v = tf.reshape(v_lin, new_shape=(1, config.s_cap, config.n_kv_heads, config.head_dim))
        q_rope, _ = tf.rope(q, q, cos_cache, sin_cache, pos_ids)
        _, k_rope = tf.rope(k, k, cos_cache, sin_cache, pos_ids)

        k_b = tf.repeat_interleave(k_rope, repeats=config.gqa_group, axis=2)
        v_b = tf.repeat_interleave(v, repeats=config.gqa_group, axis=2)
        q_h = tf.transpose(q_rope, perm=(0, 2, 1, 3))
        k_h = tf.transpose(k_b, perm=(0, 2, 1, 3))
        v_h = tf.transpose(v_b, perm=(0, 2, 1, 3))
        q_s = tf.mul(q_h, scale)
        k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
        scores = tf.add(tf.matmul(q_s, k_t), attn_mask)
        probs = tf.softmax(scores, axis=-1)
        ctx = tf.matmul(probs, v_h)
        attn_out = tf.transpose(ctx, perm=(0, 2, 1, 3))
        return tf.matmul(tf.reshape(attn_out, new_shape=(1, config.s_cap, config.q_proj)), w_o)

    @func
    def mlp(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_post: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
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
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        bias_q: Tensor[(config.q_proj,), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        bias_k: Tensor[(config.kv_proj,), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        bias_v: Tensor[(config.kv_proj,), config.dt],
        cos_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        sin_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        pos_ids: Tensor[(config.s_cap,), "i32"],
        attn_mask: Tensor[(1, 1, config.s_cap, config.s_cap), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
        gamma_post: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Full decoder layer: self_attention + residual, then mlp + residual —
        # mirrors `Qwen2DecoderLayer.forward` exactly.
        attn_out = self_attention(
            hidden, gamma_in, w_q, bias_q, w_k, bias_k, w_v, bias_v,
            cos_cache, sin_cache, pos_ids, attn_mask, scale, w_o,
        )
        h1 = tf.add(hidden, attn_out)
        mlp_out = mlp(h1, gamma_post, w_gate, w_up, w_down)
        return tf.add(h1, mlp_out)
