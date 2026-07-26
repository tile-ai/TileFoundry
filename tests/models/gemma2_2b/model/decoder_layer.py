"""Gemma-2-2B single decoder layer described as a single tilefoundry IR Module.

Phase 0 companion to ``tests/models/qwen3_1_7b/model/decoder_layer.py``: same
``@module class`` authoring style (each kernel a named ``@func`` method; the
decorator returns the ``tilefoundry.ir.core.module.Module`` the class name
binds directly to). Gemma-2's real architecture forces a different fusion
boundary than qwen3_1_7b's, though — see ``Gemma2DecoderLayer.forward``:

.. code-block:: python

    residual = hidden_states
    hidden_states = input_layernorm(hidden_states)
    hidden_states, _ = self_attn(hidden_states, ...)
    hidden_states = post_attention_layernorm(hidden_states)   # wraps ATTN OUTPUT
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = pre_feedforward_layernorm(hidden_states)
    hidden_states = mlp(hidden_states)
    hidden_states = post_feedforward_layernorm(hidden_states)  # wraps MLP OUTPUT
    hidden_states = residual + hidden_states

i.e. ``h = x + post_attn_norm(attn(input_norm(x)))``;
``out = h + post_ff_norm(mlp(pre_ff_norm(h)))``. ``post_attention_layernorm``
normalizes the *attention block's output* (pre-residual), not the next
block's input — unlike qwen3_1_7b's ``post_attention_layernorm``, which is
really a pre-MLP input norm despite the similar name. Fusing a norm into
either side of ``self_attention``/``mlp`` here would be one-sided (Gemma-2
sandwiches both blocks with norms on *both* sides), so — deliberately
different from qwen3_1_7b's self_attention/mlp, which each fuse their
preceding norm — ``self_attention`` and ``mlp`` below are pure blocks (no
norm fused in either direction; the caller must pass an already-normalized
``hidden``), and ``decoder_layer`` alone threads all four norms + both
residual adds.

Four Gemma-2-specific things to note (see ``tests/models/gemma2_2b/config.py``
module docstring for the full rundown):

- ``Gemma2RMSNorm`` is ``normed * (1.0 + weight)``; ``tf.rms_norm`` is
  ``normed * weight``. Every ``gamma*`` argument fed to ``tf.rms_norm`` below
  is expected pre-adjusted by the caller (``config.rms_gamma``) — the kernel
  stays the plain ``tf.rms_norm`` semantics throughout.
- attention scaling is ``query_pre_attn_scalar**-0.5`` (0.0625 @ 256), not
  ``head_dim**-0.5`` — passed in as the ``scale`` kernel input, same
  broadcast-scalar convention as qwen3_1_7b.
- attention logits are soft-capped before the mask is added:
  ``attn_logit_softcapping * tanh(scores / attn_logit_softcapping)``. The
  ``tf.tanh`` HIR op has no ``@register_eval`` handler (by design; not
  touched here), so this composes the identity ``tanh(z) = 2*sigmoid(2z) - 1``
  from ops that *do* have eval handlers (``sigmoid`` / ``mul`` / ``sub`` /
  ``div``), inlined directly in ``self_attention`` (the softcap is applied
  exactly once in this model, so there is no reuse to justify a separate
  helper — and a bare Python helper function couldn't be called from an
  ``@func`` body anyway: the parser resolves callees to either a
  ``tf.<op>``/``T.<op>`` schema or a sibling already-parsed ``@func``, not
  arbitrary Python).
- MLP activation is ``gelu_pytorch_tanh`` (``tf.gelu(x, approximate="tanh")``),
  not SwiGLU's ``silu`` — the one op this package's task adds to ``src/``.

GQA is 8 query / 4 kv heads (group 2, vs. qwen3_1_7b's 16/8); there is no
per-head q_norm/k_norm fused into attention (that is Qwen3-specific — Gemma-2
has none). RoPE is plain NEOX-style rotate-half, identical composition to
qwen3_1_7b's (``tf.rope`` applied on the pre-transpose ``[1,S,H,D]`` layout;
mathematically identical to HF's post-transpose ``unsqueeze_dim=1``
application since cos/sin depend only on (seq, head_dim), not the head axis).
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies

# `tf.div`/`tf.mul` take this as a value operand, and the parser only accepts a
# plain name there -- an attribute access is not a valid Expr in a @func body.
ATTN_SOFTCAP = config.attn_softcap


@module(entry="decoder_layer")
class Gemma2_2B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Pre-attention input RMSNorm; HF `Gemma2DecoderLayer.input_layernorm`.
        # `gamma_in` is pre-adjusted to `1.0 + weight` test-side.
        return tf.rms_norm(hidden, gamma_in)

    @func
    def self_attention(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        cos_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        sin_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        pos_ids: Tensor[(config.s_cap,), "i32"],
        attn_mask: Tensor[(1, 1, config.s_cap, config.s_cap), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Pure GQA + RoPE + attn-logit-softcap attention block: `hidden` is
        # already normalized (`decoder_layer` applies `input_rms_norm` before
        # calling this — see module docstring for why the norm isn't fused
        # in here, unlike qwen3_1_7b). No per-head q_norm/k_norm.
        q = tf.reshape(tf.matmul(hidden, w_q), new_shape=(1, config.s_cap, config.n_q_heads, config.head_dim))
        k = tf.reshape(tf.matmul(hidden, w_k), new_shape=(1, config.s_cap, config.n_kv_heads, config.head_dim))
        v = tf.reshape(tf.matmul(hidden, w_v), new_shape=(1, config.s_cap, config.n_kv_heads, config.head_dim))
        q_rope, _ = tf.rope(q, q, cos_cache, sin_cache, pos_ids)
        _, k_rope = tf.rope(k, k, cos_cache, sin_cache, pos_ids)

        k_b = tf.repeat_interleave(k_rope, repeats=config.gqa_group, axis=2)
        v_b = tf.repeat_interleave(v, repeats=config.gqa_group, axis=2)
        q_h = tf.transpose(q_rope, perm=(0, 2, 1, 3))
        k_h = tf.transpose(k_b, perm=(0, 2, 1, 3))
        v_h = tf.transpose(v_b, perm=(0, 2, 1, 3))
        q_s = tf.mul(q_h, scale)
        k_t = tf.transpose(k_h, perm=(0, 1, 3, 2))
        scores_raw = tf.matmul(q_s, k_t)

        # attn_logit_softcapping: config.attn_softcap * tanh(scores / config.attn_softcap),
        # applied before the mask. tanh(z) = 2*sigmoid(2z) - 1 (`tf.tanh` has
        # no eval handler; compose from ops that do).
        z = tf.div(scores_raw, ATTN_SOFTCAP)
        tanh_z = tf.sub(tf.mul(tf.sigmoid(tf.mul(z, 2.0)), 2.0), 1.0)
        scores_capped = tf.mul(tanh_z, ATTN_SOFTCAP)

        scores = tf.add(scores_capped, attn_mask)
        probs = tf.softmax(scores, axis=-1)
        ctx = tf.matmul(probs, v_h)
        attn_out = tf.transpose(ctx, perm=(0, 2, 1, 3))
        return tf.matmul(tf.reshape(attn_out, new_shape=(1, config.s_cap, config.q_proj)), w_o)

    @func
    def mlp(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Pure dense gelu_tanh-gated MLP (`hidden_activation="gelu_pytorch_tanh"`,
        # not SwiGLU's `silu`): `hidden` is already normalized
        # (`decoder_layer` applies `pre_feedforward_layernorm` first).
        gate = tf.matmul(hidden, w_gate)
        up = tf.matmul(hidden, w_up)
        act = tf.gelu(gate, approximate="tanh")
        h = tf.mul(act, up)
        return tf.matmul(h, w_down)

    @func
    def decoder_layer(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        cos_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        sin_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        pos_ids: Tensor[(config.s_cap,), "i32"],
        attn_mask: Tensor[(1, 1, config.s_cap, config.s_cap), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
        gamma_post_attn: Tensor[(config.hidden,), config.dt],
        gamma_pre_ff: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
        gamma_post_ff: Tensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # h = x + post_attn_norm(attn(input_norm(x)))
        # out = h + post_ff_norm(mlp(pre_ff_norm(h)))
        # — mirrors `Gemma2DecoderLayer.forward` exactly (all 4 norms live
        # here; self_attention / mlp are pure blocks, see module docstring).
        h_in = input_rms_norm(hidden, gamma_in)
        attn_out = self_attention(
            h_in, w_q, w_k, w_v, cos_cache, sin_cache, pos_ids, attn_mask, scale, w_o,
        )
        attn_out_n = tf.rms_norm(attn_out, gamma_post_attn)
        h1 = tf.add(hidden, attn_out_n)

        ff_in = tf.rms_norm(h1, gamma_pre_ff)
        mlp_out = mlp(ff_in, w_gate, w_up, w_down)
        mlp_out_n = tf.rms_norm(mlp_out, gamma_post_ff)
        return tf.add(h1, mlp_out_n)
