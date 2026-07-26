"""MiniCPM3-4B single decoder layer described as a single tilefoundry IR
Module — the Phase 0 **Multi-head Latent Attention (MLA)** entry.

Same ``@module class`` authoring style as ``tests/models/qwen3_1_7b/
model/decoder_layer.py`` (each kernel is a named ``@func`` method; the
decorator returns the ``tilefoundry.ir.core.module.Module`` that the class
name binds directly to). There is no MLA skeleton anywhere else in this repo
— every step below is composed from existing primitive HIR ops (``matmul``,
``rms_norm``, ``slice``, ``concat``, ``reshape``, ``transpose``, ``rope``,
``repeat_interleave``, ``softmax``, ``mul``/``add``, ``sigmoid``); no new op
was needed.

``mla_attention`` fuses the preceding ``input_layernorm`` internally (calling
the ``input_rms_norm`` kernel first, same convention as the Qwen3-1.7B
sibling's ``self_attention``); ``mlp`` fuses ``post_attention_layernorm``.
``decoder_layer`` composes ``mla_attention`` + scaled residual + ``mlp`` +
scaled residual, mirroring ``MiniCPM3DecoderLayer.forward`` exactly —
**including** the ``scale_depth`` residual scaling (MiniCPM's muP-style depth
scaling: ``hidden = residual + branch_out * residual_scale``, not a plain
add like Qwen3/Llama). ``residual_scale`` is a runtime ``Tensor[(1,1,1)]``
input (like the existing ``self_attention``'s ``scale`` tensor for attention
scaling) rather than a compile-time constant, so the HIR carries no
config-specific numeric baked in.

── MLA op-flow (``mla_attention``), matching ``MiniCPM3Attention.forward`` ──

Shapes below use ``Hq = config.n_q_heads = 40``; all tensors are batch-1,
``[1, config.s_cap, ...]`` or ``[1, config.s_cap, Hq, ...]`` layout throughout (the same
"head axis before the batch/head transpose" convention the Qwen3 siblings
use, matching what ``tf.rope`` expects) until step 7 transposes to
``[1, Hq, config.s_cap, ...]`` right before the score matmul.

1. **Q down -> norm -> up -> split** (``MiniCPM3Attention.forward`` L299-304):
   ``q_down = x @ Wq_a`` `[1,S,768]` -> ``rms_norm(., gamma_q_a, eps=1e-6)``
   -> ``q_up = . @ Wq_b`` `[1,S,Hq*96=3840]` -> reshape `[1,S,Hq,96]` ->
   slice into ``q_nope = [...,:64]`` `[1,S,Hq,64]` and
   ``q_rope = [...,64:96]`` `[1,S,Hq,32]`. The 96/64/32 split is **uneven**,
   so this uses ``tf.slice`` (``tf.split`` only supports equal-sized
   partitions — see "op-surface gotcha" below), not ``tf.split``.
2. **KV compress -> split** (L306-307): ``compressed = x @ W_kv_a_mqa``
   `[1,S,288]` -> slice into ``kv_c = [...,:256]`` `[1,S,256]` and
   ``k_rope_flat = [...,256:288]`` `[1,S,32]` (still headless — reshaped to a
   1-head 4D tensor in step 4).
3. **KV up -> split** (L309-310): ``rms_norm(kv_c, gamma_kv_a, eps=1e-6)`` ->
   ``. @ W_kv_b`` `[1,S,Hq*128=5120]` -> reshape `[1,S,Hq,128]` -> slice into
   ``k_nope = [...,:64]`` and ``value = [...,64:128]`` (each `[1,S,Hq,64]`).
   ``kv_b_proj`` up-projects the *shared* latent into one distinct
   (nope, value) pair **per query head** — this is what makes plain GQA
   head-repeat unnecessary for k_nope/value (``num_key_value_groups == 1``
   in this config); only the rope slice of K is still MQA-shared (step 5).
4. **RoPE, rope-slice only** (L312-317): ``k_rope_flat`` reshapes
   `[1,S,32]` -> `[1,S,1,32]` (a "1 shared head" axis) before
   ``tf.rope(q_rope, k_rope_4d, cos_cache, sin_cache, pos_ids)``. The cos/sin
   caches are sized **``[cache_len, qk_rope_head_dim=32]``**, not
   ``qk_head_dim=96`` — RoPE never sees the nope slice at all. ``tf.rope``'s
   typeinfer only requires ``q``/``k`` to share the *last-axis* (head_dim)
   extent (32 == 32); it does not require matching head-count, so a 40-head
   ``q_rope`` and a 1-head ``k_rope_4d`` through the same call is exactly the
   MLA / MQA-style-K use case and needs no special-casing.
5. **k_rope cross-head broadcast** (L318, ``k_rot.expand(...)``): the rotated
   1-head ``k_rope`` broadcasts to all ``Hq`` heads via
   ``tf.repeat_interleave(., repeats=config.n_q_heads, axis=2)`` — algebraically
   identical to HF's ``expand`` here because there is exactly one element
   along that axis to begin with (repeating a size-1 axis has no
   "interleave vs. broadcast" ordering ambiguity), and the same primitive the
   Qwen3 siblings use for GQA head-repeat (there the repeated axis has
   ``config.n_kv_heads`` > 1 elements; here it has exactly 1 — same op, different
   input shape).
6. **Reassemble** (L320-321): ``query = concat(q_nope, q_rope, axis=-1)``
   `[1,S,Hq,96]`, ``key = concat(k_nope, k_rope_broadcast, axis=-1)``
   `[1,S,Hq,96]` — nope first, rope second, restoring each head's original
   64/32 layout.
7. **Attention** (L333, ``eager_attention_forward``): transpose q/k/v to
   `[1,Hq,S,*]`, ``scores = (query*scaling) @ keyᵀ + causal_mask``,
   ``softmax(axis=-1)``, ``@ value`` `[1,Hq,S,64]`, transpose back, reshape
   to `[1,S,Hq*64=2560]`, ``@ Wo`` -> `[1,S,2560]`. ``scaling =
   qk_head_dim**-0.5`` (96, not 64 or 32) — read directly off
   ``layer.self_attn.scaling`` in the test, not recomputed here.

**Op-surface gotcha (no gap, but worth flagging):** HF's own split points are
uneven (``qk_nope_head_dim=64`` vs ``qk_rope_head_dim=32``;
``kv_lora_rank=256`` vs ``qk_rope_head_dim=32``) — this repo's ``Split`` HIR
op takes a ``num_splits`` *count* and requires the axis extent to divide
evenly by it (equal-sized parts only), so it cannot express a 64/32 or
256/32 partition directly. ``tf.slice`` (arbitrary per-axis
``begin``/``end``/``strides``) has no such restriction, so every split point
in this module — including the one place that happens to be even
(``qk_nope_head_dim == v_head_dim == 64`` in step 3) — uses ``tf.slice``
uniformly rather than mixing ``tf.split`` in for that one coincidentally-even
case.
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies


@module(entry="decoder_layer")
class MiniCPM3_4B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Pre-attention input RMSNorm; HF `MiniCPM3DecoderLayer.input_layernorm`
        # (eps = config.rms_norm_eps = 1e-5, config.rms_eps below — NOT the rms_norm
        # op's own 1e-6 default; see module docstring's eps gotcha).
        return tf.rms_norm(hidden, gamma_in, eps=config.rms_eps)

    @func
    def mla_attention(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q_a: Tensor[(1, config.hidden, config.q_lora_rank), config.dt],
        gamma_q_a: Tensor[(config.q_lora_rank,), config.dt],
        w_q_b: Tensor[(1, config.q_lora_rank, config.q_up_proj), config.dt],
        w_kv_a: Tensor[(1, config.hidden, config.kv_a_proj), config.dt],
        gamma_kv_a: Tensor[(config.kv_lora_rank,), config.dt],
        w_kv_b: Tensor[(1, config.kv_lora_rank, config.kv_b_proj), config.dt],
        cos_cache: Tensor[(config.s_cap, config.qk_rope_head_dim), config.dt],
        sin_cache: Tensor[(config.s_cap, config.qk_rope_head_dim), config.dt],
        pos_ids: Tensor[(config.s_cap,), "i32"],
        attn_mask: Tensor[(1, 1, config.s_cap, config.s_cap), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.attn_out, config.hidden), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Fused input_layernorm + self_attn: Multi-head Latent Attention (MLA)
        # — see the module docstring for the full 7-step breakdown. No KV
        # cache — single-shot prefill, cur_pos == 0, plain [config.s_cap,config.s_cap] mask.
        x = input_rms_norm(hidden, gamma_in)

        # Step 1: Q down -> norm -> up -> reshape -> split (nope | rope).
        q_down = tf.matmul(x, w_q_a)
        q_down_n = tf.rms_norm(q_down, gamma_q_a, eps=config.rms_eps_lora)
        q_up = tf.matmul(q_down_n, w_q_b)
        q = tf.reshape(q_up, new_shape=(1, config.s_cap, config.n_q_heads, config.qk_head_dim))
        q_nope = tf.slice(
            q, begin=(0, 0, 0, 0), end=(1, config.s_cap, config.n_q_heads, config.qk_nope_head_dim), strides=(1, 1, 1, 1)
        )
        q_rope = tf.slice(
            q, begin=(0, 0, 0, config.qk_nope_head_dim), end=(1, config.s_cap, config.n_q_heads, config.qk_head_dim),
            strides=(1, 1, 1, 1),
        )

        # Step 2: KV compress -> split (shared latent | shared rope slice).
        compressed = tf.matmul(x, w_kv_a)
        kv_c = tf.slice(
            compressed, begin=(0, 0, 0), end=(1, config.s_cap, config.kv_lora_rank), strides=(1, 1, 1)
        )
        k_rope_flat = tf.slice(
            compressed, begin=(0, 0, config.kv_lora_rank), end=(1, config.s_cap, config.kv_a_proj), strides=(1, 1, 1)
        )

        # Step 3: KV up -> reshape -> split (nope | value), one pair per head.
        kv_c_n = tf.rms_norm(kv_c, gamma_kv_a, eps=config.rms_eps_lora)
        kv_up = tf.matmul(kv_c_n, w_kv_b)
        kv = tf.reshape(kv_up, new_shape=(1, config.s_cap, config.n_q_heads, config.qk_nope_head_dim + config.v_head_dim))
        k_nope = tf.slice(
            kv, begin=(0, 0, 0, 0), end=(1, config.s_cap, config.n_q_heads, config.qk_nope_head_dim), strides=(1, 1, 1, 1)
        )
        value = tf.slice(
            kv, begin=(0, 0, 0, config.qk_nope_head_dim),
            end=(1, config.s_cap, config.n_q_heads, config.qk_nope_head_dim + config.v_head_dim), strides=(1, 1, 1, 1),
        )

        # Step 4: RoPE on the rope slice only (dim 32, not qk_head_dim 96).
        k_rope = tf.reshape(k_rope_flat, new_shape=(1, config.s_cap, 1, config.qk_rope_head_dim))
        q_rope_e, k_rope_e = tf.rope(q_rope, k_rope, cos_cache, sin_cache, pos_ids)

        # Step 5: k_rope is MQA-style shared -> broadcast to all query heads.
        k_rope_b = tf.repeat_interleave(k_rope_e, repeats=config.n_q_heads, axis=2)

        # Step 6: reassemble nope + rope (rope back in its original slot).
        query = tf.concat(q_nope, q_rope_e, axis=-1)
        key = tf.concat(k_nope, k_rope_b, axis=-1)

        # Step 7: scaled-dot-product attention + output projection.
        q_h = tf.transpose(query, perm=(0, 2, 1, 3))
        k_h = tf.transpose(key, perm=(0, 2, 1, 3))
        v_h = tf.transpose(value, perm=(0, 2, 1, 3))
        q_s = tf.mul(q_h, scale)
        k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
        scores = tf.add(tf.matmul(q_s, k_t), attn_mask)
        probs = tf.softmax(scores, axis=-1)
        ctx = tf.matmul(probs, v_h)
        attn_out = tf.transpose(ctx, perm=(0, 2, 1, 3))
        attn_flat = tf.reshape(attn_out, new_shape=(1, config.s_cap, config.attn_out))
        return tf.matmul(attn_flat, w_o)

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
        hidden_norm = tf.rms_norm(hidden, gamma_post, eps=config.rms_eps)
        gate = tf.matmul(hidden_norm, w_gate)
        up = tf.matmul(hidden_norm, w_up)
        act = tf.mul(gate, tf.sigmoid(gate))
        h = tf.mul(act, up)
        return tf.matmul(h, w_down)

    @func
    def decoder_layer(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q_a: Tensor[(1, config.hidden, config.q_lora_rank), config.dt],
        gamma_q_a: Tensor[(config.q_lora_rank,), config.dt],
        w_q_b: Tensor[(1, config.q_lora_rank, config.q_up_proj), config.dt],
        w_kv_a: Tensor[(1, config.hidden, config.kv_a_proj), config.dt],
        gamma_kv_a: Tensor[(config.kv_lora_rank,), config.dt],
        w_kv_b: Tensor[(1, config.kv_lora_rank, config.kv_b_proj), config.dt],
        cos_cache: Tensor[(config.s_cap, config.qk_rope_head_dim), config.dt],
        sin_cache: Tensor[(config.s_cap, config.qk_rope_head_dim), config.dt],
        pos_ids: Tensor[(config.s_cap,), "i32"],
        attn_mask: Tensor[(1, 1, config.s_cap, config.s_cap), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.attn_out, config.hidden), config.dt],
        gamma_post: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
        residual_scale: Tensor[(1, 1, 1), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Full decoder layer: mla_attention + scaled residual, then mlp +
        # scaled residual — mirrors `MiniCPM3DecoderLayer.forward` exactly,
        # INCLUDING the scale_depth residual scaling (MiniCPM's muP-style
        # depth scaling: `residual + branch_out * residual_scale`, not a
        # plain add like Qwen3/Llama — the one point where this decoder_layer
        # diverges structurally from the Qwen3-1.7B sibling's).
        attn_out = mla_attention(
            hidden, gamma_in, w_q_a, gamma_q_a, w_q_b, w_kv_a, gamma_kv_a, w_kv_b,
            cos_cache, sin_cache, pos_ids, attn_mask, scale, w_o,
        )
        h1 = tf.add(hidden, tf.mul(attn_out, residual_scale))
        mlp_out = mlp(h1, gamma_post, w_gate, w_up, w_down)
        return tf.add(h1, tf.mul(mlp_out, residual_scale))
