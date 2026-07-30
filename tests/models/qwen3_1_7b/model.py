"""Qwen3-1.7B's dense decoder layer and the stack that closes it, as IR Modules.
Companion to ``tests/models/qwen3_5_35b_a3b/model.py``: same
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

from tests.models.qwen3_1_7b.config import REAL as config
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.ir.types.dim import DimVar

# The prior cache this step reads: the only range this model carries. Zero is a
# first step, and the exclusive upper bound is max_ctx because a position beyond
# the rotary cache has no embedding to gather.
C = DimVar("ctx_len", 0, config.max_ctx)

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
        gamma_in: ConstTensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # Pre-attention input RMSNorm; HF `Qwen3DecoderLayer.input_layernorm`.
        return tf.rms_norm(hidden, gamma_in)

    @func
    def self_attention(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: ConstTensor[(config.hidden,), config.dt],
        w_q: ConstTensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: ConstTensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: ConstTensor[(1, config.hidden, config.kv_proj), config.dt],
        gamma_q: ConstTensor[(config.head_dim,), config.dt],
        gamma_k: ConstTensor[(config.head_dim,), config.dt],
        cos_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        sin_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        v_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: ConstTensor[(1, config.q_proj, config.hidden), config.dt],
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
        q_s = tf.reshape(q_rope, new_shape=(1, S, config.n_q_heads, config.head_dim)) * scale
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
        score_ctx = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum")
        score_new = tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum")

        # Log-sum-exp merge of the two groups' partials against their joint max.
        peak = tf.max(
            tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
        )
        peak_e = tf.reshape(peak, new_shape=(1, S, config.n_q_heads, 1, 1))
        p_ctx = tf.exp(score_ctx - peak_e)
        p_new = tf.exp(score_new - peak)
        total = tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum") + p_new
        weighted = (
            tf.reduce(p_ctx * v_ctx, axes=(-2,), keepdim=False, kind="sum")
            + p_new * v_new
        )
        attn = weighted / total
        out = tf.matmul(
            tf.reshape(attn, new_shape=(1, S, config.q_proj)), w_o
        )
        return out, k_rope, v

    @func
    def mlp(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_post: ConstTensor[(config.hidden,), config.dt],
        w_gate: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: ConstTensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # Fused post_attention_layernorm + dense SwiGLU, no residual.
        hidden_norm = tf.rms_norm(hidden, gamma_post)
        gate = tf.matmul(hidden_norm, w_gate)
        up = tf.matmul(hidden_norm, w_up)
        act = tf.silu(gate)
        h = act * up
        return tf.matmul(h, w_down)

    @func
    def tiled_mlp(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_post: ConstTensor[(config.hidden,), config.dt],
        w_gate: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: ConstTensor[(1, config.intermediate, config.hidden), config.dt],
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
            gate_z = gate_z + tf.matmul(x_k, tf.gather(wg_blk, kh, axis=0))
            up_z = up_z + tf.matmul(x_k, tf.gather(wu_blk, kh, axis=0))
        gate = tf.reshape(
            tf.transpose(gate_z, perm=(0, 2, 1, 3)), new_shape=(1, S, config.intermediate)
        )
        up = tf.reshape(
            tf.transpose(up_z, perm=(0, 2, 1, 3)), new_shape=(1, S, config.intermediate)
        )
        h = tf.silu(gate) * up
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
            out_z = out_z + tf.matmul(
                tf.gather(h_blk, ki, axis=0), tf.gather(wd_blk, ki, axis=0)
            )
        return tf.reshape(
            tf.transpose(out_z, perm=(0, 2, 1, 3)), new_shape=(1, S, config.hidden)
        )

    @func
    def decoder_layer(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: ConstTensor[(config.hidden,), config.dt],
        w_q: ConstTensor[(1, config.hidden, config.q_proj), config.dt],
        w_k: ConstTensor[(1, config.hidden, config.kv_proj), config.dt],
        w_v: ConstTensor[(1, config.hidden, config.kv_proj), config.dt],
        gamma_q: ConstTensor[(config.head_dim,), config.dt],
        gamma_k: ConstTensor[(config.head_dim,), config.dt],
        cos_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        sin_cache: Tensor[(config.max_pos, config.head_dim), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        v_cache: Tensor[(1, C, config.n_kv_heads, config.head_dim), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: ConstTensor[(1, config.q_proj, config.hidden), config.dt],
        gamma_post: ConstTensor[(config.hidden,), config.dt],
        w_gate: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: ConstTensor[(1, config.intermediate, config.hidden), config.dt],
    ):
        # One decode step: self_attention + residual, then mlp + residual --
        # mirrors `Qwen3DecoderLayer.forward` exactly -- plus this token's key
        # and value passed straight through for the caller to append.
        attn_out, k_new, v_new = self_attention(
            hidden, gamma_in, w_q, w_k, w_v, gamma_q, gamma_k,
            cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o,
        )
        h1 = hidden + attn_out
        mlp_out = mlp(h1, gamma_post, w_gate, w_up, w_down)
        return h1 + mlp_out, k_new, v_new

    # HF stores every projection as `nn.Linear.weight`, `(out, in)`; the matmuls
    # above want `(1, in, out)`. Seven separate declarations rather than one
    # mapping, because each names the published key it reads.

    @decoder_layer.converter("w_q")
    def _w_q(
        q_proj_weight: ConstTensor[(config.q_proj, config.hidden), config.dt],
    ) -> Tensor[(1, config.hidden, config.q_proj), config.dt]:
        return tf.reshape(
            tf.transpose(q_proj_weight, perm=(1, 0)),
            new_shape=(1, config.hidden, config.q_proj),
        )

    @decoder_layer.converter("w_k")
    def _w_k(
        k_proj_weight: ConstTensor[(config.kv_proj, config.hidden), config.dt],
    ) -> Tensor[(1, config.hidden, config.kv_proj), config.dt]:
        return tf.reshape(
            tf.transpose(k_proj_weight, perm=(1, 0)),
            new_shape=(1, config.hidden, config.kv_proj),
        )

    @decoder_layer.converter("w_v")
    def _w_v(
        v_proj_weight: ConstTensor[(config.kv_proj, config.hidden), config.dt],
    ) -> Tensor[(1, config.hidden, config.kv_proj), config.dt]:
        return tf.reshape(
            tf.transpose(v_proj_weight, perm=(1, 0)),
            new_shape=(1, config.hidden, config.kv_proj),
        )

    @decoder_layer.converter("w_o")
    def _w_o(
        o_proj_weight: ConstTensor[(config.hidden, config.q_proj), config.dt],
    ) -> Tensor[(1, config.q_proj, config.hidden), config.dt]:
        return tf.reshape(
            tf.transpose(o_proj_weight, perm=(1, 0)),
            new_shape=(1, config.q_proj, config.hidden),
        )

    @decoder_layer.converter("w_gate")
    def _w_gate(
        gate_proj_weight: ConstTensor[(config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, config.hidden, config.intermediate), config.dt]:
        return tf.reshape(
            tf.transpose(gate_proj_weight, perm=(1, 0)),
            new_shape=(1, config.hidden, config.intermediate),
        )

    @decoder_layer.converter("w_up")
    def _w_up(
        up_proj_weight: ConstTensor[(config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, config.hidden, config.intermediate), config.dt]:
        return tf.reshape(
            tf.transpose(up_proj_weight, perm=(1, 0)),
            new_shape=(1, config.hidden, config.intermediate),
        )

    @decoder_layer.converter("w_down")
    def _w_down(
        down_proj_weight: ConstTensor[(config.hidden, config.intermediate), config.dt],
    ) -> Tensor[(1, config.intermediate, config.hidden), config.dt]:
        return tf.reshape(
            tf.transpose(down_proj_weight, perm=(1, 0)),
            new_shape=(1, config.intermediate, config.hidden),
        )


@module
class Qwen3_1_7B_Decoder:
    """The ordered layer stack plus the norm that closes it."""

    layers = tuple(
        Qwen3_1_7B.renamed(f"layer{index}")
        for index in range(config.n_layers)
    )

    @func
    def embed(
        w_embed: ConstTensor[(config.vocab, config.hidden), config.dt],
        token_ids: Tensor[(S,), "i64"],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # HF `Qwen3Model.embed_tokens`.
        return tf.reshape(
            tf.gather(w_embed, token_ids, axis=0), new_shape=(1, S, config.hidden)
        )

    @func
    def final_rms_norm(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_final: ConstTensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # HF `Qwen3Model.norm`, applied once after the last layer.
        return tf.rms_norm(hidden, gamma_final)

    @func
    def lm_head(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        w_head: ConstTensor[(config.hidden, config.vocab), config.dt],
    ) -> Tensor[(1, config.vocab), config.dt]:
        return tf.matmul(tf.reshape(hidden, new_shape=(1, config.hidden)), w_head)

    @lm_head.converter("w_head")
    def _(
        head_weight_raw: ConstTensor[(config.vocab, config.hidden), config.dt],
    ) -> Tensor[(config.hidden, config.vocab), config.dt]:
        # HF stores the head as (vocab, hidden); the matmul above wants it the
        # other way. Tied models alias this input to the embedding table.
        return tf.transpose(head_weight_raw, perm=(1, 0))

    def forward(self, token_ids, cos_cache, sin_cache, pos_ids, scale, caches):
        """The whole decode step: this token's row, every layer over it, its logits.

        What comes back is the logits and each layer's own fresh entry; growing the
        cache with them is the caller's step, through `append_cache`.
        """
        hidden = self.embed(token_ids)
        normed, entries = self.decode_hidden(
            hidden, cos_cache, sin_cache, pos_ids, scale, caches
        )
        return self.lm_head(normed), entries

    def decode_hidden(self, hidden, cos_cache, sin_cache, pos_ids, scale, caches):
        """One decode step through every layer, then the final norm.

        *caches* is one layer's context per layer, in layer order. What comes back
        is the normalised hidden state and each layer's own cache entry, for the
        caller to append -- the same division the single layer makes.
        """
        if len(caches) != len(self.modules):
            raise ValueError(
                f"decoder has {len(self.modules)} layers but was given "
                f"{len(caches)} caches"
            )
        entries = []
        for layer, (k_cache, v_cache) in zip(self.modules, caches):
            hidden, k_new, v_new = layer(
                hidden, cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale
            )
            entries.append((k_new, v_new))
        return self.final_rms_norm(hidden), tuple(entries)

    def append_cache(self, caches, fresh):
        """The cache the next step reads: each layer's context with this step's own
        key and value written after it.

        A step hands back its own entry rather than the grown cache, so appending is
        the caller's, and the caller of a step is this root -- stated here once so a
        caller has none of its own.
        """
        import torch  # noqa: PLC0415

        return tuple(
            (torch.cat([k_cache, k_new], dim=1), torch.cat([v_cache, v_new], dim=1))
            for (k_cache, v_cache), (k_new, v_new) in zip(caches, fresh)
        )

    def init_caches(self, device="cuda"):
        """The per-layer cache container, zero positions long.

        `ctx_len` admits 0, so these are a decode start: the first step of a
        sequence attends the one position it brings itself.
        """
        import torch  # noqa: PLC0415

        from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415
        from tilefoundry.ir.types import DType  # noqa: PLC0415

        empty = (1, 0, config.n_kv_heads, config.head_dim)
        dtype = to_torch_dtype(DType.from_name(config.dt))
        return tuple(
            (
                torch.zeros(empty, device=device, dtype=dtype),
                torch.zeros(empty, device=device, dtype=dtype),
            )
            for _ in range(config.n_layers)
        )
