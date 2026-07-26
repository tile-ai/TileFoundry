"""Qwen3-1.7B dense decoder layer as one tilefoundry IR Module, over a free
``config`` name -- not importable on its own, load it with
``tests.models.loader.load_model`` (see ``../decoder_layer.py``).

Phase 0 companion to ``tests/models/qwen3_5_30b_a3b/qwen3_module.py``: same
``@module class`` authoring style (each kernel is a named ``@func`` method;
the decorator returns the ``tilefoundry.ir.core.module.Module`` that the class
name binds directly to — ``Qwen3_1_7B.lookup("self_attention")`` resolves one
kernel to its IR node). Two things make this contract simpler
than the MoE-30B one:

- the MLP is a single dense SwiGLU expert (plain gate/up/down projection), not
  the 30B sibling's runtime top-k expert routing (no router, no ``topk``, no
  ``gather``);
- there is no KV cache: this is a single-shot prefill oracle over a static
  ``config.s_cap``-token sequence (``cur_pos`` is always 0), so every kernel takes
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
directly on the ``[1, config.s_cap, heads, head_dim]`` tensor — the same reshape the
head split already produces — reproduces HF's
``q_norm(q_proj(x).view(hidden_shape))`` with no extra reshape either side of
the norm (the Qwen3 HF docstring notes exactly this: "unlike olmo, only on
the head dim... thus post q_norm does not need reshape").
"""
from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies

# ── tiled_mlp block shape ───────────────────────────────────────────────
# The AMX f32 register files (Apple M2 Pro, target/hardware/*.toml): Z holds
# 4096 B = 32x32 f32, X and Y 512 B each. MT x NT is therefore the largest
# accumulator block that stays Z-resident for a whole K walk, and one k step
# stages MT (resp. NT) f32 = 128 B in X (resp. Y). KT = 64 matches the PoC's
# measured best BM/BN/BK.
MT, NT, KT = 32, 32, 64
MB = config.s_cap // MT                 # token blocks
NB_INT = config.intermediate // NT      # gate/up column blocks
NB_HID = config.hidden // NT            # down-projection column blocks
NK_HID = config.hidden // KT            # gate/up K steps
NK_INT = config.intermediate // KT      # down-projection K steps


@module(entry="decoder_layer")
class Qwen3_1_7B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Pre-attention input RMSNorm; HF `Qwen3DecoderLayer.input_layernorm`.
        return tf.rms_norm(hidden, gamma_in)

    @func
    def self_attention(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        gamma_q: Tensor[(config.head_dim,), config.dt],
        gamma_k: Tensor[(config.head_dim,), config.dt],
        cos_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        sin_cache: Tensor[(config.s_cap, config.head_dim), config.dt],
        pos_ids: Tensor[(config.s_cap,), "i32"],
        attn_mask: Tensor[(1, 1, config.s_cap, config.s_cap), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: Tensor[(1, config.q_proj, config.hidden), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
        # Fused input_layernorm + self_attn: GQA + RoPE + per-head q_norm/k_norm,
        # no residual (the layer owns the residual add). Single-shot prefill —
        # no KV cache — so the mask is plain [config.s_cap, config.s_cap] causal (cur_pos==0).
        hidden_norm = input_rms_norm(hidden, gamma_in)
        q = tf.reshape(tf.matmul(hidden_norm, w_q), new_shape=(1, config.s_cap, config.n_q_heads, config.head_dim))
        k = tf.reshape(tf.matmul(hidden_norm, w_k), new_shape=(1, config.s_cap, config.n_kv_heads, config.head_dim))
        v = tf.reshape(tf.matmul(hidden_norm, w_v), new_shape=(1, config.s_cap, config.n_kv_heads, config.head_dim))
        q_n = tf.rms_norm(q, gamma_q)
        k_n = tf.rms_norm(k, gamma_k)
        q_rope, _ = tf.rope(q_n, q_n, cos_cache, sin_cache, pos_ids)
        _, k_rope = tf.rope(k_n, k_n, cos_cache, sin_cache, pos_ids)

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
    def tiled_mlp(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_post: Tensor[(config.hidden,), config.dt],
        w_gate: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: Tensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: Tensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, config.s_cap, config.hidden), config.dt]:
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
            tf.transpose(gate_z, perm=(0, 2, 1, 3)), new_shape=(1, config.s_cap, config.intermediate)
        )
        up = tf.reshape(
            tf.transpose(up_z, perm=(0, 2, 1, 3)), new_shape=(1, config.s_cap, config.intermediate)
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
            tf.transpose(out_z, perm=(0, 2, 1, 3)), new_shape=(1, config.s_cap, config.hidden)
        )

    @func
    def decoder_layer(
        hidden: Tensor[(1, config.s_cap, config.hidden), config.dt],
        gamma_in: Tensor[(config.hidden,), config.dt],
        w_q: Tensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: Tensor[(1, config.hidden, config.kv_proj), config.dt],
        gamma_q: Tensor[(config.head_dim,), config.dt],
        gamma_k: Tensor[(config.head_dim,), config.dt],
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
        # mirrors `Qwen3DecoderLayer.forward` exactly.
        attn_out = self_attention(
            hidden, gamma_in, w_q, w_k, w_v, gamma_q, gamma_k,
            cos_cache, sin_cache, pos_ids, attn_mask, scale, w_o,
        )
        h1 = tf.add(hidden, attn_out)
        mlp_out = mlp(h1, gamma_post, w_gate, w_up, w_down)
        return tf.add(h1, mlp_out)
