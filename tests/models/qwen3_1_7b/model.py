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
``tests/fixtures/placed/gqa_decode.py``'s
combine kernel uses. No mask is needed: a single query at the end of the
context may attend every position there is.

``self_attention`` and ``mlp`` each fuse their preceding RMSNorm internally
(``input_rms_norm`` / the post-attention norm) -- matching the Qwen3-30B-A3B
sibling's convention (its ``self_attention`` fuses ``input_rms_norm``; its
``moe`` fuses the post-attention norm) so each fused kernel lines up with one
HF pre-norm-then-block composition. ``decoder_layer`` composes
``self_attention`` + residual + ``mlp`` + residual, mirroring
``Qwen3DecoderLayer.forward`` exactly.

Every RMSNorm here is written out rather than calling ``tf.rms_norm``:
``Qwen3RMSNorm`` rounds the normalised activation to the input dtype *before* the
learned scale multiplies it, where the generic op matches ``torch.nn.RMSNorm``
and stays in f32 through that multiply.

``q_norm`` / ``k_norm`` are the same normalisation over the ``head_dim`` axis
alone, applied per head, so they run directly on the ``[1, 1, heads, head_dim]``
tensor the head split already produces.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from transformers import Qwen3Config

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403 — bare op bindings for @func bodies
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


def published(path: Path | None = None) -> Qwen3Config:
    """The checkpoint's own configuration, read by the class Hugging Face uses."""
    path = Path(__file__).parent / "config.json" if path is None else path
    return Qwen3Config(**json.loads(path.read_text(encoding="utf-8")))


config = published()

# The published dtype as the DSL spells it. The checkpoint stores its weights at
# this precision, so it is what a kernel reading them consumes.
_DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
    str(config.dtype).removeprefix("torch.")
]
_Q_PROJ = config.num_attention_heads * config.head_dim
_KV_PROJ = config.num_key_value_heads * config.head_dim
_GQA = config.num_attention_heads // config.num_key_value_heads

#: The variance floor every one of this model's norms adds, read from the
#: checkpoint rather than written down: the norms below are spelled out, so the
#: epsilon `tf.rms_norm` would have carried as an attribute is carried here.
_EPS = config.rms_norm_eps

# The prior cache this step reads: the only range this model carries. Zero is a
# first step, and the exclusive upper bound is `config.max_position_embeddings`
# because a position beyond
# the rotary cache has no embedding to gather.
C = DimVar("ctx_len", 0, config.max_position_embeddings)

# One token per step.
S = 1

_G = _GQA


def _generation_device(device):
    import torch  # noqa: PLC0415

    return torch.accelerator.current_accelerator() if device is None else device


@lru_cache(maxsize=None)
def _generation_rope(device):
    import torch  # noqa: PLC0415

    dim = config.head_dim
    inverse = 1.0 / (
        config.rope_parameters["rope_theta"]
        ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    phases = torch.outer(
        torch.arange(config.max_position_embeddings, device=device, dtype=torch.float32), inverse
    )
    phases = torch.cat((phases, phases), dim=-1)
    return phases.cos().to(config.dtype), phases.sin().to(config.dtype)


@module(entry="decoder_layer")
class Qwen3_1_7B_DecoderLayer:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        gamma_in: ConstTensor[(config.hidden_size,), _DT],
    ) -> Tensor[(1, S, config.hidden_size), _DT]:
        # HF `Qwen3DecoderLayer.input_layernorm`. Written out because
        # `Qwen3RMSNorm` ends `self.weight * hidden_states.to(input_dtype)`: it
        # rounds to the input dtype before the scale multiplies, where
        # `tf.rms_norm` keeps f32 through that multiply.
        x32 = tf.cast(hidden, dtype="f32")
        variance = tf.reduce(x32 * x32, axes=(-1,), keepdim=True, kind="mean")
        normed = tf.cast(x32 * tf.rsqrt(variance + _EPS), dtype=_DT)
        return normed * gamma_in

    @func
    def self_attention(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        gamma_in: ConstTensor[(config.hidden_size,), _DT],
        w_q: ConstTensor[(1, config.hidden_size, _Q_PROJ), _DT],
        w_k: ConstTensor[(1, config.hidden_size, _KV_PROJ), _DT],
        w_v: ConstTensor[(1, config.hidden_size, _KV_PROJ), _DT],
        gamma_q: ConstTensor[(config.head_dim,), _DT],
        gamma_k: ConstTensor[(config.head_dim,), _DT],
        cos_cache: Tensor[(config.max_position_embeddings, config.head_dim), _DT],
        sin_cache: Tensor[(config.max_position_embeddings, config.head_dim), _DT],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.num_key_value_heads, config.head_dim), _DT],
        v_cache: Tensor[(1, C, config.num_key_value_heads, config.head_dim), _DT],
        scale: Tensor[(1, 1, 1, 1), _DT],
        w_o: ConstTensor[(1, _Q_PROJ, config.hidden_size), _DT],
    ):
        # Fused input_layernorm + self_attn, no residual (the layer owns the
        # residual add). Returns the attention output together with this token's
        # key and value, which are what the caller appends to the cache.
        hidden_norm = input_rms_norm(hidden, gamma_in)
        q = tf.reshape(
            tf.matmul(hidden_norm, w_q),
            new_shape=(1, S, config.num_attention_heads, config.head_dim),
        )
        k = tf.reshape(
            tf.matmul(hidden_norm, w_k),
            new_shape=(1, S, config.num_key_value_heads, config.head_dim),
        )
        v = tf.reshape(
            tf.matmul(hidden_norm, w_v),
            new_shape=(1, S, config.num_key_value_heads, config.head_dim),
        )
        q_n32 = tf.cast(q, dtype="f32")
        q_n_var = tf.reduce(q_n32 * q_n32, axes=(-1,), keepdim=True, kind="mean")
        q_n = tf.cast(q_n32 * tf.rsqrt(q_n_var + _EPS), dtype=_DT) * gamma_q
        q_rope, _ = tf.rope(q_n, q_n, cos_cache, sin_cache, pos_ids)
        k_n32 = tf.cast(k, dtype="f32")
        k_n_var = tf.reduce(k_n32 * k_n32, axes=(-1,), keepdim=True, kind="mean")
        k_n = tf.cast(k_n32 * tf.rsqrt(k_n_var + _EPS), dtype=_DT) * gamma_k
        _, k_rope = tf.rope(k_n, k_n, cos_cache, sin_cache, pos_ids)

        # Every query head sees its group's key/value head, for the cache and
        # for the new token alike.
        q_s = tf.reshape(q_rope, new_shape=(1, S, config.num_attention_heads, config.head_dim)) * scale
        k_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(k_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, config.num_attention_heads, C, config.head_dim),
        )
        v_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(v_cache, repeats=_G, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, config.num_attention_heads, C, config.head_dim),
        )
        k_new = tf.repeat_interleave(k_rope, repeats=_G, axis=2)
        v_new = tf.repeat_interleave(v, repeats=_G, axis=2)

        # Two score groups: one over the cache, one over the token itself.
        q_e = tf.reshape(q_s, new_shape=(1, S, config.num_attention_heads, 1, config.head_dim))
        score_ctx = tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum")
        score_new = tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum")

        # Log-sum-exp merge of the two groups' partials against their joint max.
        peak = tf.max(
            tf.reduce(score_ctx, axes=(-2,), keepdim=False, kind="max"), score_new
        )
        peak_e = tf.reshape(peak, new_shape=(1, S, config.num_attention_heads, 1, 1))
        p_ctx = tf.exp(score_ctx - peak_e)
        p_new = tf.exp(score_new - peak)
        total = tf.reduce(p_ctx, axes=(-2,), keepdim=False, kind="sum") + p_new
        weighted = (
            tf.reduce(p_ctx * v_ctx, axes=(-2,), keepdim=False, kind="sum")
            + p_new * v_new
        )
        attn = weighted / total
        out = tf.matmul(
            tf.reshape(attn, new_shape=(1, S, _Q_PROJ)), w_o
        )
        return out, k_rope, v

    @func
    def mlp(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        gamma_post: ConstTensor[(config.hidden_size,), _DT],
        w_gate: ConstTensor[(1, config.hidden_size, config.intermediate_size), _DT],
        w_up: ConstTensor[(1, config.hidden_size, config.intermediate_size), _DT],
        w_down: ConstTensor[(1, config.intermediate_size, config.hidden_size), _DT],
    ) -> Tensor[(1, S, config.hidden_size), _DT]:
        # Fused post_attention_layernorm + dense SwiGLU, no residual.
        hidden_norm32 = tf.cast(hidden, dtype="f32")
        hidden_norm_var = tf.reduce(hidden_norm32 * hidden_norm32, axes=(-1,), keepdim=True, kind="mean")
        hidden_norm = tf.cast(hidden_norm32 * tf.rsqrt(hidden_norm_var + _EPS), dtype=_DT) * gamma_post
        gate = tf.matmul(hidden_norm, w_gate)
        up = tf.matmul(hidden_norm, w_up)
        act = tf.silu(gate)
        h = act * up
        return tf.matmul(h, w_down)

    @func
    def decoder_layer(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        gamma_in: ConstTensor[(config.hidden_size,), _DT],
        w_q: ConstTensor[(1, config.hidden_size, _Q_PROJ), _DT],
        w_k: ConstTensor[(1, config.hidden_size, _KV_PROJ), _DT],
        w_v: ConstTensor[(1, config.hidden_size, _KV_PROJ), _DT],
        gamma_q: ConstTensor[(config.head_dim,), _DT],
        gamma_k: ConstTensor[(config.head_dim,), _DT],
        cos_cache: Tensor[(config.max_position_embeddings, config.head_dim), _DT],
        sin_cache: Tensor[(config.max_position_embeddings, config.head_dim), _DT],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.num_key_value_heads, config.head_dim), _DT],
        v_cache: Tensor[(1, C, config.num_key_value_heads, config.head_dim), _DT],
        scale: Tensor[(1, 1, 1, 1), _DT],
        w_o: ConstTensor[(1, _Q_PROJ, config.hidden_size), _DT],
        gamma_post: ConstTensor[(config.hidden_size,), _DT],
        w_gate: ConstTensor[(1, config.hidden_size, config.intermediate_size), _DT],
        w_up: ConstTensor[(1, config.hidden_size, config.intermediate_size), _DT],
        w_down: ConstTensor[(1, config.intermediate_size, config.hidden_size), _DT],
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


@module(target=CudaTarget("nvidia.h200_sxm"), topologies=(Topology("cta", 132), Topology("thread", 512)))
class Qwen3_1_7B:
    """The ordered layer stack plus the norm that closes it."""


    layers = tuple(
        Qwen3_1_7B_DecoderLayer.renamed(f"layer{index}")
        for index in range(config.num_hidden_layers)
    )

    @func
    def embed(
        w_embed: ConstTensor[(config.vocab_size, config.hidden_size), _DT],
        token_ids: Tensor[(S,), "i64"],
    ) -> Tensor[(1, S, config.hidden_size), _DT]:
        # HF `Qwen3Model.embed_tokens`.
        return tf.reshape(
            tf.gather(w_embed, token_ids, axis=0), new_shape=(1, S, config.hidden_size)
        )

    @func
    def final_rms_norm(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        gamma_final: ConstTensor[(config.hidden_size,), _DT],
    ) -> Tensor[(1, S, config.hidden_size), _DT]:
        # HF `Qwen3Model.norm`, applied once after the last layer.
        out32 = tf.cast(hidden, dtype="f32")
        out_var = tf.reduce(out32 * out32, axes=(-1,), keepdim=True, kind="mean")
        out = tf.cast(out32 * tf.rsqrt(out_var + _EPS), dtype=_DT) * gamma_final
        return out

    @func
    def lm_head(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        w_head: ConstTensor[(config.hidden_size, config.vocab_size), _DT],
    ) -> Tensor[(1, config.vocab_size), _DT]:
        return tf.matmul(tf.reshape(hidden, new_shape=(1, config.hidden_size)), w_head)

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

    def init_caches(self, device=None):
        """The per-layer cache container, zero positions long.

        `ctx_len` admits 0, so these are a decode start: the first step of a
        sequence attends the one position it brings itself.
        """
        import torch  # noqa: PLC0415

        from tilefoundry.evaluator.value import to_torch_dtype  # noqa: PLC0415
        from tilefoundry.ir.types import DType  # noqa: PLC0415

        device = _generation_device(device)
        empty = (1, 0, config.num_key_value_heads, config.head_dim)
        dtype = to_torch_dtype(DType.from_name(_DT))
        return tuple(
            (
                torch.zeros(empty, device=device, dtype=dtype),
                torch.zeros(empty, device=device, dtype=dtype),
            )
            for _ in range(config.num_hidden_layers)
        )

    def prepare_inputs_for_generation(self, input_ids, step, caches, device=None):
        """The token and positional activations for one decode step."""
        import torch  # noqa: PLC0415

        device = _generation_device(device)
        token_ids = input_ids[step].reshape(1).to(device=device, dtype=torch.int64)
        cos_cache, sin_cache = _generation_rope(device)
        pos_ids = torch.tensor([step], device=device, dtype=torch.int32)
        scale = torch.full(
            (1, 1, 1, 1), config.head_dim ** -0.5, device=device, dtype=config.dtype
        )
        return token_ids, cos_cache, sin_cache, pos_ids, scale, caches
