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

Decode, one token per step. The step's own token count is the literal 1, so the
only dimension carried as a range is the context the step reads: ``ctx_len``,
the length of the KV cache handed in.

The cache is explicit tensors in and out, and the two directions are not the
same tensor. What comes in is the context *before* this token -- ``ctx_len``
positions, read-only. What goes out is this token's own key and value, one
position each. Appending the second to the first is the caller's step, not the
kernel's, and that is what keeps every shape here expressed in ``ctx_len``
alone: a kernel returning the grown cache would have an axis of ``ctx_len + 1``,
and a sum of a range and a constant cannot feed the matmul that would consume it.

That split is also why attention here is an online softmax rather than one
``softmax`` over a concatenated score row. The new token has to attend to itself
as well as to the cache, and the two score groups live in differently shaped
tensors; each is reduced to its own ``(max, sum, weighted values)`` partial and
the partials are merged by a log-sum-exp rescale. No mask is needed: a single
query at the end of the context may attend every position there is -- which is
also why Gemma-2's alternating sliding-window layers do not appear here. A
window only removes positions from the front of the context, so for a context no
longer than ``sliding_window`` a sliding layer and a full layer are the same
computation; ``config.max_ctx`` is pinned to ``sliding_window`` so that stays
true rather than being assumed (see ``config.py``).

Four Gemma-2-specific things to note (see ``tests/models/gemma2_2b/config.py``
module docstring for the full rundown):

- ``Gemma2RMSNorm`` is ``normed * (1.0 + weight)``; ``tf.rms_norm`` is
  ``normed * weight``. Every ``gamma*`` argument fed to ``tf.rms_norm`` below
  is expected pre-adjusted by the caller (``config.rms_gamma``) — the kernel
  stays the plain ``tf.rms_norm`` semantics throughout.
- attention scaling is ``query_pre_attn_scalar**-0.5`` (0.0625 @ 256), not
  ``head_dim**-0.5`` — passed in as the ``scale`` kernel input, same
  broadcast-scalar convention as qwen3_1_7b.
- attention logits are soft-capped, on the raw scaled scores and before
  anything reduces them:
  ``attn_logit_softcapping * tanh(scores / attn_logit_softcapping)``. The
  ``tf.tanh`` HIR op has no ``@register_eval`` handler (by design; not
  touched here), so this composes the identity ``tanh(z) = 2*sigmoid(2z) - 1``
  from ops that *do* have eval handlers (``sigmoid`` / ``mul`` / ``sub`` /
  ``div``). Written out once per score group inside ``self_attention`` rather
  than factored out: the two groups are differently shaped, a ``@func`` binds
  its parameter shapes exactly (``hir.function.elaborate``), and a plain Python
  helper is not a callee the parser resolves at all — it accepts a
  ``tf.<op>``/``T.<op>`` schema or a sibling already-parsed ``@func``, nothing
  else.
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
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.types.dim import DimVar

# The active context length: the only range this model carries. DimVar bounds are
# half-open, so the envelope's exclusive upper bound admits max_ctx itself. The
# lower bound is 1: a step with no prior context is a prefill, not a decode step.
C = DimVar("ctx_len", 1, config.max_ctx + 1)

# One token per step.
S = 1

# `tf.div`/`tf.mul` take this as a value operand, and the parser only accepts a
# plain name there -- an attribute access is not a valid Expr in a @func body.
ATTN_SOFTCAP = config.attn_softcap


@module(entry="decoder_layer")
class Gemma2_2B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # Pre-attention input RMSNorm; HF `Gemma2DecoderLayer.input_layernorm`.
        # `gamma_in` is pre-adjusted to `1.0 + weight` test-side.
        return tf.rms_norm(hidden, gamma_in)

    @func
    def self_attention(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        cos_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        sin_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        v_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
    ):
        # Pure GQA + RoPE + attn-logit-softcap attention block: `hidden` is
        # already normalized (`decoder_layer` applies `input_rms_norm` before
        # calling this — see module docstring for why the norm isn't fused
        # in here, unlike qwen3_1_7b). No per-head q_norm/k_norm.
        #
        # Returns the attention output with this token's key and value, which are
        # what the caller appends to the cache.
        q = tf.reshape(tf.matmul(hidden, w_q), new_shape=(1, S, config.n_q_heads, config.head_dim))
        k = tf.reshape(tf.matmul(hidden, w_k), new_shape=(1, S, config.n_kv_heads, config.head_dim))
        v = tf.reshape(tf.matmul(hidden, w_v), new_shape=(1, S, config.n_kv_heads, config.head_dim))
        q_rope, _ = tf.rope(q, q, cos_cache, sin_cache, pos_ids)
        _, k_rope = tf.rope(k, k, cos_cache, sin_cache, pos_ids)

        # Every query head sees its group's key/value head, for the cache and for
        # the new token alike. No mask: one query at the end of the context may
        # attend every position there is.
        q_s = tf.mul(q_rope, scale)
        k_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(k_cache, repeats=config.gqa_group, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, config.n_q_heads, C, config.head_dim),
        )
        v_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(v_cache, repeats=config.gqa_group, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, config.n_q_heads, C, config.head_dim),
        )
        k_new = tf.repeat_interleave(k_rope, repeats=config.gqa_group, axis=2)
        v_new = tf.repeat_interleave(v, repeats=config.gqa_group, axis=2)

        # Two score groups: one over the cache, one over the token itself, each
        # soft-capped on its own raw logits -- `cap * tanh(score / cap)`, with
        # tanh composed as `2*sigmoid(2z) - 1` because `tf.tanh` carries no
        # evaluation handler. The cap is elementwise on a logit, so it goes
        # before the maximum, where `eager_attention_forward` puts it; capping
        # after the merge would cap a normalisation instead. Spelled out for both
        # groups rather than shared, because the two are differently shaped: a
        # @func binds its parameter shapes exactly, and a plain Python helper is
        # not a callee the @func parser resolves.
        q_e = tf.reshape(q_s, new_shape=(1, S, config.n_q_heads, 1, config.head_dim))
        z_ctx = tf.div(
            tf.reduce(tf.mul(q_e, k_ctx), axes=(-1,), keepdim=True, kind="sum"), ATTN_SOFTCAP
        )
        score_ctx = tf.mul(
            tf.sub(tf.mul(tf.sigmoid(tf.mul(z_ctx, 2.0)), 2.0), 1.0), ATTN_SOFTCAP
        )
        z_new = tf.div(
            tf.reduce(tf.mul(q_s, k_new), axes=(-1,), keepdim=True, kind="sum"), ATTN_SOFTCAP
        )
        score_new = tf.mul(
            tf.sub(tf.mul(tf.sigmoid(tf.mul(z_new, 2.0)), 2.0), 1.0), ATTN_SOFTCAP
        )

        # Log-sum-exp merge of the two groups against their joint max.
        peak = tf.max(
            tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
        )
        peak_e = tf.reshape(peak, new_shape=(1, S, config.n_q_heads, 1, 1))
        p_ctx = tf.exp(tf.sub(score_ctx, peak_e))
        p_new = tf.exp(tf.sub(score_new, peak))
        total = tf.add(tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum"), p_new)
        weighted = tf.add(
            tf.reduce(tf.mul(p_ctx, v_ctx), axes=(-2,), keepdim=False, kind="sum"),
            tf.mul(p_new, v_new),
        )
        attn = tf.div(weighted, total)
        out = tf.matmul(tf.reshape(attn, new_shape=(1, S, config.q_proj)), w_o)
        return out, k_rope, v

    @func
    def mlp(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
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
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        cos_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        sin_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        v_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
        gamma_post_attn: Tensor[(config.hidden,), config.dt],
        gamma_pre_ff: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
        gamma_post_ff: Tensor[(config.hidden,), config.dt],
    ):
        # h = x + post_attn_norm(attn(input_norm(x)))
        # out = h + post_ff_norm(mlp(pre_ff_norm(h)))
        # — mirrors `Gemma2DecoderLayer.forward` exactly (all 4 norms live
        # here; self_attention / mlp are pure blocks, see module docstring).
        h_in = input_rms_norm(hidden, gamma_in)
        attn_out, k_new, v_new = self_attention(
            h_in, w_q, w_k, w_v, cos_cache, sin_cache, pos_ids,
            k_cache, v_cache, scale, w_o,
        )
        attn_out_n = tf.rms_norm(attn_out, gamma_post_attn)
        h1 = tf.add(hidden, attn_out_n)

        ff_in = tf.rms_norm(h1, gamma_pre_ff)
        mlp_out = mlp(ff_in, w_gate, w_up, w_down)
        mlp_out_n = tf.rms_norm(mlp_out, gamma_post_ff)
        return tf.add(h1, mlp_out_n), k_new, v_new
