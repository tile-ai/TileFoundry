"""Qwen3.5-35B-A3B's ``full_attention`` token mixer as one tilefoundry IR Module,
over a free ``config`` name -- not importable on its own, load it with
``tests.models.loader.load_model`` (see ``../full_attention.py``).

One layer in four is this one; the other three are the Gated DeltaNet in
``linear_attention.py``. What makes this boundary its own rather than a copy of
Qwen3's GQA are two things the published configuration states and Qwen3 does not
have:

- **a partial rotary embedding.** ``partial_rotary_factor`` is 0.25, so of each
  head's 256 entries only the first 64 rotate and the remaining 192 pass through
  carrying no position at all. The rotary caches are therefore 64 wide, not 256,
  and the kernel has to split each head, rotate one part and concatenate the
  other back -- which is exactly what Hugging Face's own ``apply_rotary_pos_emb``
  does when it slices ``q`` to ``cos.shape[-1]``.
- **an output gate.** ``q_proj`` fans out to *two* ``head_dim`` blocks per query
  head. One is the query; the other is a gate that the attention output is
  multiplied by, through a sigmoid, before ``o_proj``. So the projection is
  8192 wide where a Qwen3 layer's would be 4096, and half of it never reaches a
  score.

  Hugging Face applies this gate unconditionally -- it does not read
  ``attn_output_gate`` at all; the flag being true in the published
  configuration agrees with the code rather than selecting it. A fixture that
  branched on the flag would be describing a model Hugging Face cannot build.

Decode, one token per step. The step's own token count is the literal 1, so the
only dimension carried as a range is ``ctx_len``, the length of the KV cache
handed in. The cache is explicit tensors in and out, and the two directions are
not the same tensor: what comes in is the context *before* this token,
``ctx_len`` positions, read-only; what goes out is this token's own key and
value, one position each. Appending is the caller's step, which is what keeps
every shape here expressed in ``ctx_len`` alone.

Attention is an online softmax over two groups -- the cache and the token
itself -- merged by a log-sum-exp rescale against their joint max. No mask is
needed: a single query at the end of the context may attend every position
there is.
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 -- tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 -- bare op bindings
from tilefoundry.ir.types.dim import DimVar

# The active context length: the only range this kernel carries. DimVar bounds
# are half-open [lo, hi), so the envelope's exclusive upper bound is max_ctx + 1
# to keep the largest supported context inside it. The lower bound is 1: a step
# with no prior context is a prefill, not a decode step.
C = DimVar("ctx_len", 1, config.max_ctx + 1)

# One token per step.
S = 1

_H = config.hidden
_HQ = config.n_q_heads
_HKV = config.n_kv_heads
_D = config.head_dim
_ROT = config.rotary_dim
_PASS = config.pass_dim
_G = config.gqa_group

# The rotary caches need one row per position a step may be decoded at, which is
# at most ``max_ctx``; ``max_position_embeddings`` is 262144 and a cache that
# size is 67 MB of zeros nothing reads.
_ROPE_ROWS = config.max_ctx + 1


@module(entry="full_attention")
class Qwen3_5FullAttention:
    @func
    def partial_rope(
        x: Tensor[(1, S, _HQ, _D), config.dt],
        cos_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        sin_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        pos_ids: Tensor[(S,), "i32"],
    ) -> Tensor[(1, S, _HQ, _D), config.dt]:
        # Rotate the leading `rotary_dim` of each head and concatenate the
        # untouched tail back on. `tf.rope` multiplies its caches against the
        # whole of its input's last axis, so the split is what makes a partial
        # factor expressible at all rather than an optional rearrangement.
        rot = tf.slice(
            x, begin=(0, 0, 0, 0), end=(1, S, _HQ, _ROT), strides=(1, 1, 1, 1)
        )
        tail = tf.slice(
            x, begin=(0, 0, 0, _ROT), end=(1, S, _HQ, _D), strides=(1, 1, 1, 1)
        )
        turned, _ = tf.rope(rot, rot, cos_cache, sin_cache, pos_ids)
        return tf.concat(turned, tail, axis=-1)

    @func
    def partial_rope_kv(
        x: Tensor[(1, S, _HKV, _D), config.dt],
        cos_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        sin_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        pos_ids: Tensor[(S,), "i32"],
    ) -> Tensor[(1, S, _HKV, _D), config.dt]:
        # The same rotation over the key's head count. Its own Function because a
        # Function's parameter shapes are fixed and GQA's two head counts differ.
        rot = tf.slice(
            x, begin=(0, 0, 0, 0), end=(1, S, _HKV, _ROT), strides=(1, 1, 1, 1)
        )
        tail = tf.slice(
            x, begin=(0, 0, 0, _ROT), end=(1, S, _HKV, _D), strides=(1, 1, 1, 1)
        )
        turned, _ = tf.rope(rot, rot, cos_cache, sin_cache, pos_ids)
        return tf.concat(turned, tail, axis=-1)

    @func
    def full_attention(
        hidden: Tensor[(1, S, _H), config.dt],
        gamma_in: Tensor[(_H,), config.dt],
        w_qg: Tensor[(1, _H, _HQ * _D * 2), config.dt],
        w_k: Tensor[(1, _H, _HKV * _D), config.dt],
        w_v: Tensor[(1, _H, _HKV * _D), config.dt],
        gamma_q: Tensor[(_D,), config.dt],
        gamma_k: Tensor[(_D,), config.dt],
        cos_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        sin_cache: Tensor[(_ROPE_ROWS, _ROT), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, _HKV, _D), config.dt],
        v_cache: Tensor[(1, C, _HKV, _D), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, _HQ * _D, _H), config.dt],
    ):
        # Fused input_layernorm + `Qwen3_5MoeAttention`, no residual (the layer
        # owns the residual add). Returns the attention output together with this
        # token's key and value, which are what the caller appends to the cache.
        hidden_norm = tf.rms_norm(hidden, gamma_in)

        # One projection, two halves: the query and the output gate. The split is
        # over the last axis of the [heads, 2 * head_dim] view, so gate entry j of
        # head h sits beside query entry j of the same head, not in a second
        # contiguous block of the flat projection.
        qg = tf.reshape(
            tf.matmul(hidden_norm, w_qg), new_shape=(1, S, _HQ, 2 * _D)
        )
        q = tf.slice(qg, begin=(0, 0, 0, 0), end=(1, S, _HQ, _D), strides=(1, 1, 1, 1))
        gate = tf.slice(
            qg, begin=(0, 0, 0, _D), end=(1, S, _HQ, 2 * _D), strides=(1, 1, 1, 1)
        )

        q_rope = partial_rope(
            tf.rms_norm(q, gamma_q), cos_cache, sin_cache, pos_ids
        )
        k_rope = partial_rope_kv(
            tf.rms_norm(
                tf.reshape(tf.matmul(hidden_norm, w_k), new_shape=(1, S, _HKV, _D)),
                gamma_k,
            ),
            cos_cache, sin_cache, pos_ids,
        )
        v = tf.reshape(tf.matmul(hidden_norm, w_v), new_shape=(1, S, _HKV, _D))

        # Every query head sees its group's key/value head, for the cache and for
        # the new token alike.
        q_s = tf.mul(q_rope, scale)
        k_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(k_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, _HQ, C, _D),
        )
        v_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(v_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, _HQ, C, _D),
        )
        k_new = tf.repeat_interleave(k_rope, repeats=_G, axis=2)
        v_new = tf.repeat_interleave(v, repeats=_G, axis=2)

        # Two score groups: one over the cache, one over the token itself.
        q_e = tf.reshape(q_s, new_shape=(1, S, _HQ, 1, _D))
        score_ctx = tf.reduce(tf.mul(q_e, k_ctx), axes=(-1,), keepdim=True, kind="sum")
        score_new = tf.reduce(tf.mul(q_s, k_new), axes=(-1,), keepdim=True, kind="sum")

        # Log-sum-exp merge of the two groups' partials against their joint max.
        peak = tf.max(
            tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
        )
        peak_e = tf.reshape(peak, new_shape=(1, S, _HQ, 1, 1))
        p_ctx = tf.exp(tf.sub(score_ctx, peak_e))
        p_new = tf.exp(tf.sub(score_new, peak))
        total = tf.add(tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum"), p_new)
        weighted = tf.add(
            tf.reduce(tf.mul(p_ctx, v_ctx), axes=(-2,), keepdim=False, kind="sum"),
            tf.mul(p_new, v_new),
        )
        attn = tf.div(weighted, total)

        # The output gate, then o_proj. Head-major flattening on both sides, so
        # gate entry (h, j) meets attention entry (h, j).
        gated = tf.mul(
            tf.reshape(attn, new_shape=(1, S, _HQ * _D)),
            tf.sigmoid(tf.reshape(gate, new_shape=(1, S, _HQ * _D))),
        )
        return tf.matmul(gated, w_o), k_rope, v
