"""Qwen3-1.7B dense decoder layer as one tilefoundry IR Module, over a free
``config`` name -- not importable on its own, load it with
``tests.models.loader.load_model`` (see ``../decoder_layer.py``).

Companion to ``tests/models/qwen3_5_35b_a3b/model/``: same
``@module class`` authoring style (each kernel is a named ``@func`` method; the
decorator returns the ``tilefoundry.ir.core.module.Module`` that the class name
binds directly to -- ``Qwen3_1_7B.lookup("self_attention")`` resolves one kernel
to its IR node). What differs from the MoE-30B sibling is the MLP: a single
dense SwiGLU expert (plain gate/up/down projection), with none of the 30B's
runtime top-k expert routing -- no router, no ``topk``, no ``gather``.

Decode, one token per step. The step's own token count is the literal 1, so the
only dimension carried as a range is the context the step reads: ``ctx_len``,
the length of the KV cache handed in. Everything a caller has to know how to
compute is that one number.

The cache is explicit tensors in and out, and the two directions are not the
same tensor. What comes in is the context *before* this token -- ``ctx_len``
positions, read-only. What goes out is this token's own key and value, one
position each. Appending the second to the first is the caller's step, not the
kernel's, and that is what keeps every shape here expressed in ``ctx_len``
alone: a kernel returning the grown cache would have an axis of ``ctx_len + 1``,
and a sum of a range and a constant cannot feed the matmul that would consume it
(the constraint that makes a step return its own entry rather than the grown
cache).

That split is also why attention here is an online softmax rather than one
``softmax`` over a concatenated score row. The new token has to attend to itself
as well as to the cache, and the two score groups live in differently shaped
tensors; each is reduced to its own ``(max, sum, weighted values)`` partial and
the partials are merged by the same log-sum-exp rescale
``tests/fixtures/gqa_online.py``'s
combine kernel uses. No mask is needed: a single query at the end of the
context may attend every position there is.

``self_attention`` and ``mlp`` each fuse their preceding RMSNorm internally
(``input_rms_norm`` / the post-attention norm) -- matching the Qwen3-30B-A3B
sibling's convention (its ``self_attention`` fuses ``input_rms_norm``; its
``moe`` fuses the post-attention norm) so each fused kernel lines up with one
HF pre-norm-then-block composition. ``decoder_layer`` composes
``self_attention`` + residual + ``mlp`` + residual, mirroring
``Qwen3DecoderLayer.forward`` exactly.

Qwen3's per-head ``q_norm`` / ``k_norm`` (RMSNorm over just the ``head_dim``
axis, applied to every head independently) needs no special HIR combinator:
``tf.rms_norm`` normalizes only the last axis and is rank-agnostic on every
axis before it (see ``tilefoundry/ir/hir/nn/rms_norm.py``), so calling it
directly on the ``[1, 1, heads, head_dim]`` tensor -- the same reshape the head
split already produces -- reproduces HF's ``q_norm(q_proj(x).view(hidden_shape))``
with no extra reshape either side of the norm (the Qwen3 HF docstring notes
exactly this: "unlike olmo, only on the head dim... thus post q_norm does not
need reshape").
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.ir.types.dim import DimVar

# The active context length: the only range this model carries. DimVar bounds
# are half-open [lo, hi), so the envelope's exclusive upper bound is max_ctx + 1
# to keep the largest supported context inside it. The lower bound is 1: a step
# with no prior context is a prefill, not a decode step.
C = DimVar("ctx_len", 1, config.max_ctx + 1)

# One token per step.
S = 1

# ── tiled_mlp block shape ───────────────────────────────────────────────
# The AMX f32 register files (Apple M2 Pro, target/hardware/*.toml): Z holds
# 4096 B = 32x32 f32, X and Y 512 B each. NT x KT are sized against those; the
# token axis is not blocked at all, because a decode step has one token to
# block. MT = 1 makes each block matmul the [1, KT] @ [KT, NT] row-times-panel
# the step actually performs.
MT, NT, KT = 1, 32, 64
MB = S // MT                            # token blocks
NB_INT = config.intermediate // NT      # gate/up column blocks
NB_HID = config.hidden // NT            # down-projection column blocks
NK_HID = config.hidden // KT            # gate/up K steps
NK_INT = config.intermediate // KT      # down-projection K steps

_G = config.gqa_group


@module(entry="decoder_layer")
class Qwen3_1_7B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # Pre-attention input RMSNorm; HF `Qwen3DecoderLayer.input_layernorm`.
        return tf.rms_norm(hidden, gamma_in)

    @func
    def self_attention(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        gamma_q: Tensor[(config.head_dim,), config.dt],
        gamma_k: Tensor[(config.head_dim,), config.dt],
        cos_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        sin_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        v_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
    ):
        # Fused input_layernorm + self_attn, no residual (the layer owns the
        # residual add). Returns the attention output together with this token's
        # key and value, which are what the caller appends to the cache.
        hidden_norm = input_rms_norm(hidden, gamma_in)
        q = tf.reshape(
            tf.matmul(hidden_norm, w_q),
            new_shape=(1, S, config.n_q_heads, config.head_dim),
        )
        k = tf.reshape(
            tf.matmul(hidden_norm, w_k),
            new_shape=(1, S, config.n_kv_heads, config.head_dim),
        )
        v = tf.reshape(
            tf.matmul(hidden_norm, w_v),
            new_shape=(1, S, config.n_kv_heads, config.head_dim),
        )
        q_n = tf.rms_norm(q, gamma_q)
        q_rope, _ = tf.rope(q_n, q_n, cos_cache, sin_cache, pos_ids)
        k_n = tf.rms_norm(k, gamma_k)
        _, k_rope = tf.rope(k_n, k_n, cos_cache, sin_cache, pos_ids)

        # Every query head sees its group's key/value head, for the cache and
        # for the new token alike.
        q_s = tf.mul(tf.reshape(q_rope, new_shape=(1, S, config.n_q_heads, config.head_dim)), scale)
        k_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(k_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, config.n_q_heads, C, config.head_dim),
        )
        v_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(v_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, config.n_q_heads, C, config.head_dim),
        )
        k_new = tf.repeat_interleave(k_rope, repeats=_G, axis=2)
        v_new = tf.repeat_interleave(v, repeats=_G, axis=2)

        # Two score groups: one over the cache, one over the token itself.
        q_e = tf.reshape(q_s, new_shape=(1, S, config.n_q_heads, 1, config.head_dim))
        score_ctx = tf.reduce(tf.mul(q_e, k_ctx), axes=(-1,), keepdim=True, kind="sum")
        score_new = tf.reduce(tf.mul(q_s, k_new), axes=(-1,), keepdim=True, kind="sum")

        # Log-sum-exp merge of the two groups' partials against their joint max.
        peak = tf.max(
            tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
        )
        peak_e = tf.reshape(peak, new_shape=(1, S, config.n_q_heads, 1, 1))
        p_ctx = tf.exp(tf.sub(score_ctx, peak_e))
        p_new = tf.exp(tf.sub(score_new, peak))
        total = tf.add(
            tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum"), p_new
        )
        weighted = tf.add(
            tf.reduce(tf.mul(p_ctx, v_ctx), axes=(-2,), keepdim=False, kind="sum"),
            tf.mul(p_new, v_new),
        )
        attn = tf.div(weighted, total)
        out = tf.matmul(
            tf.reshape(attn, new_shape=(1, S, config.q_proj)), w_o
        )
        return out, k_rope, v

    @func
    def mlp(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_post: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # Fused post_attention_layernorm + dense SwiGLU, no residual. silu(x) =
        # x * sigmoid(x) — there is no standalone silu op in the HIR op surface.
        hidden_norm = tf.rms_norm(hidden, gamma_post)
        gate = tf.matmul(hidden_norm, w_gate)
        up = tf.matmul(hidden_norm, w_up)
        act = tf.mul(gate, tf.sigmoid(gate))
        h = tf.mul(act, up)
        return tf.matmul(h, w_down)

    @func
    def tiled_mlp(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_post: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # Same value as `mlp`, written as the loop nest AMX wants: every matmul
        # is [MT, KT] @ [KT, NT] over a (token-block, column-block) batch pair,
        # and the K walk is an authored `for ... in tile(...)` whose carried arg
        # IS the accumulator buffer — `zeros` declares it, the loop carry holds
        # it, and nothing allocates. The reshape/transpose pairs only re-index:
        # [1, S, K] -> [NK, MB, 1, MT, KT] blocks the M/K axes, [1, K, N] ->
        # [NK, 1, NB, KT, NT] the K/N axes, and `gather(_, k, axis=0)` picks
        # iteration k's K slice of both.
        hidden_norm = tf.rms_norm(hidden, gamma_post)
        x_blk = tf.reshape(
            tf.transpose(
                tf.reshape(hidden_norm, new_shape=(MB, MT, NK_HID, KT)), perm=(2, 0, 1, 3)
            ),
            new_shape=(NK_HID, MB, 1, MT, KT),
        )
        wg_blk = tf.reshape(
            tf.transpose(
                tf.reshape(w_gate, new_shape=(NK_HID, KT, NB_INT, NT)), perm=(0, 2, 1, 3)
            ),
            new_shape=(NK_HID, 1, NB_INT, KT, NT),
        )
        wu_blk = tf.reshape(
            tf.transpose(
                tf.reshape(w_up, new_shape=(NK_HID, KT, NB_INT, NT)), perm=(0, 2, 1, 3)
            ),
            new_shape=(NK_HID, 1, NB_INT, KT, NT),
        )
        gate_z = tf.zeros(shape=(MB, NB_INT, MT, NT), dtype=config.dt)
        up_z = tf.zeros(shape=(MB, NB_INT, MT, NT), dtype=config.dt)
        for kh in tile(NK_HID):
            x_k = tf.gather(x_blk, kh, axis=0)
            gate_z = tf.add(gate_z, tf.matmul(x_k, tf.gather(wg_blk, kh, axis=0)))
            up_z = tf.add(up_z, tf.matmul(x_k, tf.gather(wu_blk, kh, axis=0)))
        gate = tf.reshape(
            tf.transpose(gate_z, perm=(0, 2, 1, 3)), new_shape=(1, S, config.intermediate)
        )
        up = tf.reshape(
            tf.transpose(up_z, perm=(0, 2, 1, 3)), new_shape=(1, S, config.intermediate)
        )
        h = tf.mul(tf.mul(gate, tf.sigmoid(gate)), up)
        h_blk = tf.reshape(
            tf.transpose(tf.reshape(h, new_shape=(MB, MT, NK_INT, KT)), perm=(2, 0, 1, 3)),
            new_shape=(NK_INT, MB, 1, MT, KT),
        )
        wd_blk = tf.reshape(
            tf.transpose(
                tf.reshape(w_down, new_shape=(NK_INT, KT, NB_HID, NT)), perm=(0, 2, 1, 3)
            ),
            new_shape=(NK_INT, 1, NB_HID, KT, NT),
        )
        out_z = tf.zeros(shape=(MB, NB_HID, MT, NT), dtype=config.dt)
        for ki in tile(NK_INT):
            out_z = tf.add(
                out_z,
                tf.matmul(tf.gather(h_blk, ki, axis=0), tf.gather(wd_blk, ki, axis=0)),
            )
        return tf.reshape(
            tf.transpose(out_z, perm=(0, 2, 1, 3)), new_shape=(1, S, config.hidden)
        )

    @func
    def decoder_layer(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        gamma_q: Tensor[(config.head_dim,), config.dt],
        gamma_k: Tensor[(config.head_dim,), config.dt],
        cos_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        sin_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        v_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
        gamma_post: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
    ):
        # One decode step: self_attention + residual, then mlp + residual --
        # mirrors `Qwen3DecoderLayer.forward` exactly -- plus this token's key
        # and value passed straight through for the caller to append.
        attn_out, k_new, v_new = self_attention(
            hidden, gamma_in, w_q, w_k, w_v, gamma_q, gamma_k,
            cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o,
        )
        h1 = tf.add(hidden, attn_out)
        mlp_out = mlp(h1, gamma_post, w_gate, w_up, w_down)
        return tf.add(h1, mlp_out), k_new, v_new
