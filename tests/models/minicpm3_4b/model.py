"""MiniCPM3-4B's dense decoder layer and the stack that closes it, as IR Modules.

The corpus's only **Multi-head Latent Attention (MLA)** model, in the same
``@module class`` authoring style as ``tests/models/qwen2_5_1_5b/model.py``: each kernel is a named ``@func`` method and the decorator
returns the ``tilefoundry.ir.core.module.Module`` the class name binds to, so
``MiniCPM3_4B.lookup("mla_attention")`` resolves one kernel to its IR node. Every
step is composed from primitive HIR ops; MLA needed no new one.

Decode, one token per step. The step's own token count is the literal 1, so the
only dimension carried as a range is the context the step reads: ``ctx_len``, the
length of the KV cache handed in.

The cache is explicit tensors in and out, and the two directions are not the same
tensor. What comes in is the context *before* this token -- ``ctx_len``
positions, read-only. What goes out is this token's own key and value, one
position each. Appending the second to the first is the caller's step, which is
what keeps every shape here expressed in ``ctx_len`` alone: a kernel returning
the grown cache would have an axis of ``ctx_len + 1``, and a sum of a range and a
constant cannot feed the matmul that would consume it.

That split is also why attention is an online softmax rather than one ``softmax``
over a concatenated score row. The new token attends itself as well as the cache,
the two score groups live in differently shaped tensors, and each is reduced to
its own ``(max, sum, weighted values)`` partial before a log-sum-exp rescale
merges them. No mask is needed: a single query at the end of the context may
attend every position there is.

── What the caches hold ─────────────────────────────────────────────────────

``k_cache`` is the assembled per-head key, ``[1, ctx_len, heads, 96]``, and
``v_cache`` the up-projected value, ``[1, ctx_len, heads, 64]`` -- not the 288-wide
latent a production MLA stack would cache. That is Hugging Face's own cache
content (``MiniCPM3Attention.forward`` calls ``past_key_values.update`` after
assembling both), and matching it is what lets the oracle be exact without
constructing a ``Cache``; ``../config.py`` states the evidence. Two things follow
in the signatures below: the key cache and the value cache have different head
dims, and ``num_key_value_heads == num_attention_heads``, so nothing repeats the
cache across heads on the way in.

── The step, matching ``MiniCPM3Attention.forward`` ─────────────────────────

1. **Q down -> norm -> up -> split**: ``x @ Wq_a`` `[1,1,768]` ->
   ``rms_norm(., gamma_q_a, eps=1e-6)`` -> ``@ Wq_b`` `[1,1,40*96]` -> reshape
   `[1,1,40,96]` -> ``q_nope = [...,:64]``, ``q_rope = [...,64:]``. The split is
   uneven, so it is a subscript; this repo's ``Split`` op takes a count and
   requires equal parts, so it cannot express 64/32 (or 256/32) at all, and every
   split here is subscripted uniformly rather than mixing the two.
2. **KV compress -> split**: ``x @ W_kv_a_mqa`` `[1,1,288]` ->
   ``kv_c = [...,:256]``, ``k_rope_flat = [...,256:]`` (headless; one shared
   rotary slice for all 40 heads).
3. **KV up -> split**: ``rms_norm(kv_c, gamma_kv_a, eps=1e-6) @ W_kv_b``
   `[1,1,40*128]` -> reshape `[1,1,40,128]` -> ``k_nope = [...,:64]``,
   ``value = [...,64:]``. The up-projection produces one distinct (nope, value)
   pair *per query head*, which is why plain GQA head-repeat is unnecessary for
   them.
4. **RoPE, rotary slice only**: ``k_rope_flat`` reshapes to `[1,1,1,32]` (a "one
   shared head" axis) before ``tf.rope(q_rope, k_rope, ...)``. The cos/sin caches
   are `[max_pos, 32]`, never the full 96 -- RoPE does not see the nope slice.
   ``tf.rope`` only requires its two operands to share the last-axis extent, not
   the head count, so 40-head Q and 1-head K through one call is exactly this
   MQA-style use.
5. **Broadcast the rotary slice** across all query heads with
   ``tf.repeat_interleave(..., repeats=n_q_heads, axis=2)`` -- algebraically
   identical to HF's ``expand`` because the axis has exactly one element, so
   there is no interleave-versus-broadcast ordering to get wrong.
6. **Reassemble**: ``query = concat(q_nope, q_rope)``, ``key = concat(k_nope,
   k_rope_broadcast)``, nope first, restoring each head's 64/32 layout.
7. **Attend** the cache and the token itself, online-softmax merged, then
   ``@ Wo``. ``scaling = qk_head_dim ** -0.5`` (96, not 64 or 32) arrives as the
   ``scale`` tensor, read off ``layer.self_attn.scaling`` by the test rather than
   recomputed here.

``mla_attention`` fuses the preceding ``input_layernorm`` and ``mlp`` the
post-attention one, so each fused kernel lines up with one HF
pre-norm-then-block composition. ``decoder_layer`` composes ``mla_attention`` +
scaled residual + ``mlp`` + scaled residual, mirroring
``MiniCPM3DecoderLayer.forward`` -- **including** the ``scale_depth`` residual
scaling (``residual + branch * residual_scale``, not the plain add the Qwen
siblings make). ``residual_scale`` is a runtime ``Tensor[(1,1,1)]`` like the
attention ``scale``, so the HIR carries no config-specific number baked in --
which also keeps it correct for a stack of any depth, since the scale divides by
``sqrt(num_hidden_layers)``.

The root brackets that stack with the two scalars MiniCPM3 puts at the model's
ends: ``embed`` multiplies the gathered row by ``scale_emb`` (12, HF's
``MiniCPM3ScaledWordEmbedding``), and ``lm_head`` divides the hidden state by
``logits_scaling`` (10.0) before the matmul, where ``MiniCPM3ForCausalLM``
divides it. Both are config-derived constants rather than runtime tensors like
``residual_scale``: neither depends on the depth.
"""
from __future__ import annotations

from tests.models.minicpm3_4b.config import REAL as config
from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.ir.types.dim import DimVar

# The active context length: the only range this model carries. DimVar bounds
# are half-open [lo, hi), so the envelope's exclusive upper bound is max_ctx + 1
# to keep the largest supported context inside it. The lower bound is 1: a step
# with no prior context is a prefill, not a decode step.
C = DimVar("ctx_len", 1, config.max_ctx + 1)

# One token per step.
S = 1

_H = config.n_q_heads
_QK = config.qk_head_dim
_NOPE = config.qk_nope_head_dim
_V = config.v_head_dim
_KV_PAIR = config.qk_nope_head_dim + config.v_head_dim

EMBED_SCALE = float(config.scale_emb)
LOGITS_SCALING = config.logits_scaling


@module(entry="decoder_layer")
class MiniCPM3_4B:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: ConstTensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # Pre-attention input RMSNorm; HF `MiniCPM3DecoderLayer.input_layernorm`
        # (eps = config.rms_norm_eps = 1e-5, NOT the rms_norm op's own 1e-6
        # default, which is what the two low-rank norms below use).
        return tf.rms_norm(hidden, gamma_in, eps=config.rms_eps)

    @func
    def mla_attention(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: ConstTensor[(config.hidden,), config.dt],
        w_q_a: ConstTensor[(1, config.hidden, config.q_lora_rank), config.dt],
        gamma_q_a: ConstTensor[(config.q_lora_rank,), config.dt],
        w_q_b: ConstTensor[(1, config.q_lora_rank, config.q_up_proj), config.dt],
        w_kv_a: ConstTensor[(1, config.hidden, config.kv_a_proj), config.dt],
        gamma_kv_a: ConstTensor[(config.kv_lora_rank,), config.dt],
        w_kv_b: ConstTensor[(1, config.kv_lora_rank, config.kv_b_proj), config.dt],
        cos_cache: Tensor[(config.max_pos, config.qk_rope_head_dim), config.dt],
        sin_cache: Tensor[(config.max_pos, config.qk_rope_head_dim), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.n_kv_heads, config.qk_head_dim), config.dt],
        v_cache: Tensor[(1, C, config.n_kv_heads, config.v_head_dim), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: ConstTensor[(1, config.attn_out, config.hidden), config.dt],
    ):
        # Fused input_layernorm + MLA self_attn, no residual (the layer owns the
        # residual add). Returns the attention output together with this token's
        # assembled key and value, which are what the caller appends.
        x = input_rms_norm(hidden, gamma_in)

        # Step 1: Q down -> norm -> up -> reshape -> split (nope | rope).
        q_down = tf.matmul(x, w_q_a)
        q_up = tf.matmul(tf.rms_norm(q_down, gamma_q_a, eps=config.rms_eps_lora), w_q_b)
        q = tf.reshape(q_up, new_shape=(1, S, _H, _QK))
        q_nope = q[:, :, :, :_NOPE]
        q_rope = q[:, :, :, _NOPE:_QK]

        # Step 2: KV compress -> split (shared latent | shared rotary slice).
        compressed = tf.matmul(x, w_kv_a)
        kv_c = compressed[:, :, : config.kv_lora_rank]
        k_rope_flat = compressed[:, :, config.kv_lora_rank : config.kv_a_proj]

        # Step 3: KV up -> reshape -> split (nope | value), one pair per head.
        kv_up = tf.matmul(tf.rms_norm(kv_c, gamma_kv_a, eps=config.rms_eps_lora), w_kv_b)
        kv = tf.reshape(kv_up, new_shape=(1, S, _H, _KV_PAIR))
        k_nope = kv[:, :, :, :_NOPE]
        v_new = kv[:, :, :, _NOPE:_KV_PAIR]

        # Step 4: RoPE on the rotary slice only (dim 32, not qk_head_dim 96).
        k_rope = tf.reshape(k_rope_flat, new_shape=(1, S, 1, config.qk_rope_head_dim))
        q_rope_e, k_rope_e = tf.rope(q_rope, k_rope, cos_cache, sin_cache, pos_ids)

        # Step 5: the rotary slice of K is MQA-shared -> broadcast to every head.
        k_rope_b = tf.repeat_interleave(k_rope_e, repeats=_H, axis=2)

        # Step 6: reassemble nope + rope, each back in its original slot.
        query = tf.concat(q_nope, q_rope_e, axis=-1)
        k_new = tf.concat(k_nope, k_rope_b, axis=-1)

        # Step 7: attend the cache and the token itself, then project out.
        q_s = query * scale
        k_ctx = tf.reshape(
            tf.transpose(k_cache, perm=(0, 2, 1, 3)), new_shape=(1, 1, _H, C, _QK)
        )
        v_ctx = tf.reshape(
            tf.transpose(v_cache, perm=(0, 2, 1, 3)), new_shape=(1, 1, _H, C, _V)
        )

        # Two score groups: one over the cache, one over the token itself.
        q_e = tf.reshape(q_s, new_shape=(1, S, _H, 1, _QK))
        score_ctx = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum")
        score_new = tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum")

        # Log-sum-exp merge of the two groups' partials against their joint max.
        peak = tf.max(
            tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
        )
        peak_e = tf.reshape(peak, new_shape=(1, S, _H, 1, 1))
        p_ctx = tf.exp(score_ctx - peak_e)
        p_new = tf.exp(score_new - peak)
        total = tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum") + p_new
        weighted = (
            tf.reduce(p_ctx * v_ctx, axes=(-2,), keepdim=False, kind="sum")
            + p_new * v_new
        )
        attn = weighted / total
        out = tf.matmul(tf.reshape(attn, new_shape=(1, S, config.attn_out)), w_o)
        return out, k_new, v_new

    @func
    def mlp(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_post: ConstTensor[(config.hidden,), config.dt],
        w_gate: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: ConstTensor[(1, config.intermediate, config.hidden), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # Fused post_attention_layernorm + dense SwiGLU, no residual.
        hidden_norm = tf.rms_norm(hidden, gamma_post, eps=config.rms_eps)
        gate = tf.matmul(hidden_norm, w_gate)
        up = tf.matmul(hidden_norm, w_up)
        act = tf.silu(gate)
        h = act * up
        return tf.matmul(h, w_down)

    @func
    def decoder_layer(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_in: ConstTensor[(config.hidden,), config.dt],
        w_q_a: ConstTensor[(1, config.hidden, config.q_lora_rank), config.dt],
        gamma_q_a: ConstTensor[(config.q_lora_rank,), config.dt],
        w_q_b: ConstTensor[(1, config.q_lora_rank, config.q_up_proj), config.dt],
        w_kv_a: ConstTensor[(1, config.hidden, config.kv_a_proj), config.dt],
        gamma_kv_a: ConstTensor[(config.kv_lora_rank,), config.dt],
        w_kv_b: ConstTensor[(1, config.kv_lora_rank, config.kv_b_proj), config.dt],
        cos_cache: Tensor[(config.max_pos, config.qk_rope_head_dim), config.dt],
        sin_cache: Tensor[(config.max_pos, config.qk_rope_head_dim), config.dt],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.n_kv_heads, config.qk_head_dim), config.dt],
        v_cache: Tensor[(1, C, config.n_kv_heads, config.v_head_dim), config.dt],
        scale: Tensor[(1, 1, 1, 1), config.dt],
        w_o: ConstTensor[(1, config.attn_out, config.hidden), config.dt],
        gamma_post: ConstTensor[(config.hidden,), config.dt],
        w_gate: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_up: ConstTensor[(1, config.hidden, config.intermediate), config.dt],
        w_down: ConstTensor[(1, config.intermediate, config.hidden), config.dt],
        residual_scale: Tensor[(1, 1, 1), config.dt],
    ):
        # One decode step: mla_attention + scaled residual, then mlp + scaled
        # residual -- mirrors `MiniCPM3DecoderLayer.forward` exactly, INCLUDING
        # the scale_depth residual scaling -- plus this token's key and value
        # passed straight through for the caller to append.
        attn_out, k_new, v_new = mla_attention(
            hidden, gamma_in, w_q_a, gamma_q_a, w_q_b, w_kv_a, gamma_kv_a, w_kv_b,
            cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale, w_o,
        )
        h1 = hidden + attn_out * residual_scale
        mlp_out = mlp(h1, gamma_post, w_gate, w_up, w_down)
        return h1 + mlp_out * residual_scale, k_new, v_new


@module
class MiniCPM3_4B_Decoder:
    """The ordered layer stack, the norm that closes it, and the two scaled ends
    that bracket it."""

    layers = tuple(
        MiniCPM3_4B.renamed(f"layer{index}")
        for index in range(config.n_layers)
    )

    @func
    def embed(
        w_embed: ConstTensor[(config.vocab, config.hidden), config.dt],
        token_ids: Tensor[(S,), "i64"],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # HF `MiniCPM3ScaledWordEmbedding`: scaled by `scale_emb`.
        row = tf.reshape(
            tf.gather(w_embed, token_ids, axis=0), new_shape=(1, S, config.hidden)
        )
        return row * EMBED_SCALE

    @func
    def final_rms_norm(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        gamma_final: ConstTensor[(config.hidden,), config.dt],
    ) -> Tensor[(1, S, config.hidden), config.dt]:
        # HF `MiniCPM3Model.norm`, applied once after the last layer, at
        # config.rms_norm_eps like the two norms inside a layer.
        return tf.rms_norm(hidden, gamma_final, eps=config.rms_eps)

    @func
    def lm_head(
        hidden: Tensor[(1, S, config.hidden), config.dt],
        w_head: ConstTensor[(config.hidden, config.vocab), config.dt],
    ) -> Tensor[(1, config.vocab), config.dt]:
        # `MiniCPM3ForCausalLM.forward` divides the hidden state by
        # `logits_scaling` before the head, not after.
        scaled = tf.reshape(hidden, new_shape=(1, config.hidden)) / LOGITS_SCALING
        return tf.matmul(scaled, w_head)

    @lm_head.converter("w_head")
    def _(
        head_weight_raw: ConstTensor[(config.vocab, config.hidden), config.dt],
    ) -> Tensor[(config.hidden, config.vocab), config.dt]:
        # HF stores the head as (vocab, hidden); the matmul above wants it the
        # other way. MiniCPM3 ties its head, so this input is the embedding table.
        return tf.transpose(head_weight_raw, perm=(1, 0))

    def forward(
        self, token_ids, cos_cache, sin_cache, pos_ids, scale, residual_scale, caches,
    ):
        """The whole decode step: this token's row, every layer over it, its logits.

        What comes back is the logits and each layer's own fresh entry; growing the
        cache with them is the caller's step, through `append_cache`.
        """
        hidden = self.embed(token_ids)
        normed, entries = self.decode_hidden(
            hidden, cos_cache, sin_cache, pos_ids, scale, residual_scale, caches
        )
        return self.lm_head(normed), entries

    def decode_hidden(
        self, hidden, cos_cache, sin_cache, pos_ids, scale, residual_scale, caches,
    ):
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
                hidden, cos_cache, sin_cache, pos_ids, k_cache, v_cache, scale,
                residual_scale,
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

        A container, not a decode start: no prefix produced these, and `ctx_len`
        is bounded below by 1, so `forward` needs a context the caller prefilled.
        MLA's halves differ in shape -- the key carries the nope and rope slices,
        the value only its own head dim -- so the pair is stated twice.
        """
        import torch  # noqa: PLC0415

        from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415
        from tilefoundry.ir.types import DType  # noqa: PLC0415

        dtype = to_torch_dtype(DType.from_name(config.dt))
        return tuple(
            (
                torch.zeros((1, 0, config.n_kv_heads, _QK), device=device, dtype=dtype),
                torch.zeros((1, 0, config.n_kv_heads, _V), device=device, dtype=dtype),
            )
            for _ in range(config.n_layers)
        )
