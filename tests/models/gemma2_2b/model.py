"""Gemma-2-2B's dense decoder layer and the stack that closes it, as IR Modules.

Phase 0 companion to ``tests/models/qwen3_1_7b/model.py``: same
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
computation; the kernels use that window as their context envelope so that
stays true rather than being assumed.

Six Gemma-2-specific things to note:

- ``Gemma2RMSNorm`` is ``normed * (1.0 + weight)``; ``tf.rms_norm`` is
  ``normed * weight``. The ``1.0 +`` is this model's own semantics, so every norm
  below adds it inline and a loader hands over the checkpoint's tensor untouched.
  The widen to f32 first is HF's too — ``1.0 + self.weight.float()`` — and it
  matters because adding one to a bf16 weight rounds most of the weight away.
- attention scaling is ``query_pre_attn_scalar**-0.5`` (0.0625 @ 256), not
  ``head_dim**-0.5`` — passed in as the ``scale`` kernel input, same
  broadcast-scalar convention as qwen3_1_7b.
- attention logits are soft-capped, on the raw scaled scores and before
  anything reduces them:
  ``attn_logit_softcapping * tanh(scores / attn_logit_softcapping)``. The
  Written out once per score group inside ``self_attention`` rather
  than factored out: the two groups are differently shaped, a ``@func`` binds
  its parameter shapes exactly (``hir.function.elaborate``), and a plain Python
  helper is not a callee the parser resolves at all — it accepts a
  ``tf.<op>``/``T.<op>`` schema or a sibling already-parsed ``@func``, nothing
  else.
- MLP activation is ``gelu_pytorch_tanh`` (``tf.gelu(x, approximate="tanh")``),
  not SwiGLU's ``silu``.
- the token embedding carries a scale: ``Gemma2TextScaledWordEmbedding``
  multiplies the gathered row by ``hidden_size ** 0.5`` (48.0), so ``embed`` is
  a gather *and* a multiply.
- the output logits are soft-capped as well, at ``final_logit_softcapping``
  (30.0) rather than attention's 50.0. ``lm_head`` composes its tanh by the same
  identity ``self_attention`` uses, for the same reason.

GQA is 8 query / 4 kv heads (group 2, vs. qwen3_1_7b's 16/8); there is no
per-head q_norm/k_norm fused into attention (that is Qwen3-specific — Gemma-2
has none). RoPE is plain NEOX-style rotate-half, identical composition to
qwen3_1_7b's (``tf.rope`` applied on the pre-transpose ``[1,S,H,D]`` layout;
mathematically identical to HF's post-transpose ``unsqueeze_dim=1``
application since cos/sin depend only on (seq, head_dim), not the head axis).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from transformers import Gemma2Config

from tilefoundry import func, module
from tilefoundry.dsl import ConstTensor, Tensor, tf  # noqa: F401 — tf used by @func bodies
from tilefoundry.dsl.tf import *  # noqa: F401, F403
from tilefoundry.ir.types.dim import DimVar
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget


def published(path: Path | None = None, **overrides) -> Gemma2Config:
    """The checkpoint's own configuration, read by the class Hugging Face uses.

    *overrides* are passed to the config class alongside the published fields.
    The oracle is the one caller that uses them, and it asks for
    ``attn_implementation="eager"``: Gemma-2 caps attention logits at 50.0 and
    only the eager path applies the cap, so a reference built any other way is a
    different model.
    """
    path = Path(__file__).parent / "config.json" if path is None else path
    return Gemma2Config(**{**json.loads(path.read_text(encoding="utf-8")), **overrides})


config = published()

# The published dtype as the DSL spells it.
_DT = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[
    str(config.dtype).removeprefix("torch.")
]
_Q_PROJ = config.num_attention_heads * config.head_dim
_KV_PROJ = config.num_key_value_heads * config.head_dim
_GQA = config.num_attention_heads // config.num_key_value_heads

#: `Gemma2TextScaledWordEmbedding` multiplies the gathered row by this.
_EMBED_SCALE = config.hidden_size ** 0.5

# The prior cache this step reads: the only range this model carries. Zero is a
# first step, and the exclusive upper bound is `config.sliding_window`, so with this
# step's own token the total is one `_within_window` admits.
C = DimVar("ctx_len", 0, config.sliding_window)

# One token per step.
S = 1

ATTN_SOFTCAP = config.attn_logit_softcapping
LOGIT_SOFTCAP = config.final_logit_softcapping
EMBED_SCALE = _EMBED_SCALE


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
    phases = torch.outer(torch.arange(config.sliding_window, device=device, dtype=torch.float32), inverse)
    phases = torch.cat((phases, phases), dim=-1)
    return phases.cos().to(config.dtype), phases.sin().to(config.dtype)


@module(entry="decoder_layer")
class Gemma2_2B_DecoderLayer:
    @func
    def input_rms_norm(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        gamma_in: ConstTensor[(config.hidden_size,), _DT],
    ) -> Tensor[(1, S, config.hidden_size), _DT]:
        # Pre-attention input RMSNorm; HF `Gemma2DecoderLayer.input_layernorm`.
        return tf.rms_norm(hidden, tf.cast(gamma_in, dtype="f32") + 1.0)

    @func
    def self_attention(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        w_q: ConstTensor[(1, config.hidden_size, _Q_PROJ), _DT],
        w_k: ConstTensor[(1, config.hidden_size, _KV_PROJ), _DT],
        w_v: ConstTensor[(1, config.hidden_size, _KV_PROJ), _DT],
        cos_cache: Tensor[(config.max_position_embeddings, config.head_dim), _DT],
        sin_cache: Tensor[(config.max_position_embeddings, config.head_dim), _DT],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.num_key_value_heads, config.head_dim), _DT],
        v_cache: Tensor[(1, C, config.num_key_value_heads, config.head_dim), _DT],
        scale: Tensor[(1, 1, 1, 1), _DT],
        w_o: ConstTensor[(1, _Q_PROJ, config.hidden_size), _DT],
    ):
        # Pure GQA + RoPE + attn-logit-softcap attention block: `hidden` is
        # already normalized (`decoder_layer` applies `input_rms_norm` before
        # calling this — see module docstring for why the norm isn't fused
        # in here, unlike qwen3_1_7b). No per-head q_norm/k_norm.
        #
        # Returns the attention output with this token's key and value, which are
        # what the caller appends to the cache.
        q = tf.reshape(tf.matmul(hidden, w_q), new_shape=(1, S, config.num_attention_heads, config.head_dim))
        k = tf.reshape(tf.matmul(hidden, w_k), new_shape=(1, S, config.num_key_value_heads, config.head_dim))
        v = tf.reshape(tf.matmul(hidden, w_v), new_shape=(1, S, config.num_key_value_heads, config.head_dim))
        q_rope, _ = tf.rope(q, q, cos_cache, sin_cache, pos_ids)
        _, k_rope = tf.rope(k, k, cos_cache, sin_cache, pos_ids)

        # Every query head sees its group's key/value head, for the cache and for
        # the new token alike. No mask: one query at the end of the context may
        # attend every position there is.
        q_s = q_rope * scale
        k_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(k_cache, repeats=_GQA, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, config.num_attention_heads, C, config.head_dim),
        )
        v_ctx = tf.reshape(
            tf.transpose(tf.repeat_interleave(v_cache, repeats=_GQA, axis=2), perm=(0, 2, 1, 3)),
            new_shape=(1, 1, config.num_attention_heads, C, config.head_dim),
        )
        k_new = tf.repeat_interleave(k_rope, repeats=_GQA, axis=2)
        v_new = tf.repeat_interleave(v, repeats=_GQA, axis=2)

        # Two score groups, each soft-capped on its own raw logits. The cap is
        # elementwise on a logit, so it goes before the maximum, where
        # `eager_attention_forward` puts it; capping after the merge would cap a
        # normalisation instead. Written out per group because the two are
        # differently shaped and a @func binds its parameter shapes exactly.
        q_e = tf.reshape(q_s, new_shape=(1, S, config.num_attention_heads, 1, config.head_dim))
        z_ctx = (
            tf.reduce(q_e * k_ctx, axes=(-1,), keepdim=True, kind="sum") / ATTN_SOFTCAP
        )
        score_ctx = tf.tanh(z_ctx) * ATTN_SOFTCAP
        z_new = (
            tf.reduce(q_s * k_new, axes=(-1,), keepdim=True, kind="sum") / ATTN_SOFTCAP
        )
        score_new = tf.tanh(z_new) * ATTN_SOFTCAP

        # Log-sum-exp merge of the two groups against their joint max.
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
        out = tf.matmul(tf.reshape(attn, new_shape=(1, S, _Q_PROJ)), w_o)
        return out, k_rope, v

    @func
    def mlp(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        w_gate: ConstTensor[(1, config.hidden_size, config.intermediate_size), _DT],
        w_up: ConstTensor[(1, config.hidden_size, config.intermediate_size), _DT],
        w_down: ConstTensor[(1, config.intermediate_size, config.hidden_size), _DT],
    ) -> Tensor[(1, S, config.hidden_size), _DT]:
        # Pure dense gelu_tanh-gated MLP (`hidden_activation="gelu_pytorch_tanh"`,
        # not SwiGLU's `silu`): `hidden` is already normalized
        # (`decoder_layer` applies `pre_feedforward_layernorm` first).
        gate = tf.matmul(hidden, w_gate)
        up = tf.matmul(hidden, w_up)
        act = tf.gelu(gate, approximate="tanh")
        h = act * up
        return tf.matmul(h, w_down)

    @func
    def decoder_layer(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        gamma_in: ConstTensor[(config.hidden_size,), _DT],
        w_q: ConstTensor[(1, config.hidden_size, _Q_PROJ), _DT],
        w_k: ConstTensor[(1, config.hidden_size, _KV_PROJ), _DT],
        w_v: ConstTensor[(1, config.hidden_size, _KV_PROJ), _DT],
        cos_cache: Tensor[(config.max_position_embeddings, config.head_dim), _DT],
        sin_cache: Tensor[(config.max_position_embeddings, config.head_dim), _DT],
        pos_ids: Tensor[(S,), "i32"],
        k_cache: Tensor[(1, C, config.num_key_value_heads, config.head_dim), _DT],
        v_cache: Tensor[(1, C, config.num_key_value_heads, config.head_dim), _DT],
        scale: Tensor[(1, 1, 1, 1), _DT],
        w_o: ConstTensor[(1, _Q_PROJ, config.hidden_size), _DT],
        gamma_post_attn: ConstTensor[(config.hidden_size,), _DT],
        gamma_pre_ff: ConstTensor[(config.hidden_size,), _DT],
        w_gate: ConstTensor[(1, config.hidden_size, config.intermediate_size), _DT],
        w_up: ConstTensor[(1, config.hidden_size, config.intermediate_size), _DT],
        w_down: ConstTensor[(1, config.intermediate_size, config.hidden_size), _DT],
        gamma_post_ff: ConstTensor[(config.hidden_size,), _DT],
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
        attn_out_n = tf.rms_norm(attn_out, tf.cast(gamma_post_attn, dtype="f32") + 1.0)
        h1 = hidden + attn_out_n

        ff_in = tf.rms_norm(h1, tf.cast(gamma_pre_ff, dtype="f32") + 1.0)
        mlp_out = mlp(ff_in, w_gate, w_up, w_down)
        mlp_out_n = tf.rms_norm(mlp_out, tf.cast(gamma_post_ff, dtype="f32") + 1.0)
        return h1 + mlp_out_n, k_new, v_new


@module(target=CudaTarget("nvidia.h200_sxm"))
class Gemma2_2B:
    """The ordered layer stack, the norm that closes it, and the scaled embedding
    and soft-capped head that bracket it."""

    topologies = (Topology("cta", 132), Topology("thread", 512))

    layers = tuple(
        Gemma2_2B_DecoderLayer.renamed(f"layer{index}")
        for index in range(config.num_hidden_layers)
    )

    @func
    def embed(
        w_embed: ConstTensor[(config.vocab_size, config.hidden_size), _DT],
        token_ids: Tensor[(S,), "i64"],
    ) -> Tensor[(1, S, config.hidden_size), _DT]:
        # HF `Gemma2TextScaledWordEmbedding`: scaled by `hidden_size ** 0.5`.
        row = tf.reshape(
            tf.gather(w_embed, token_ids, axis=0), new_shape=(1, S, config.hidden_size)
        )
        return row * EMBED_SCALE

    @func
    def final_rms_norm(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        gamma_final: ConstTensor[(config.hidden_size,), _DT],
    ) -> Tensor[(1, S, config.hidden_size), _DT]:
        # HF `Gemma2Model.norm`, applied once after the last layer.
        return tf.rms_norm(hidden, tf.cast(gamma_final, dtype="f32") + 1.0)

    @func
    def lm_head(
        hidden: Tensor[(1, S, config.hidden_size), _DT],
        w_head: ConstTensor[(config.hidden_size, config.vocab_size), _DT],
    ) -> Tensor[(1, config.vocab_size), _DT]:
        # Soft-capped as `Gemma2ForCausalLM.forward` caps it, at
        # `final_logit_softcapping` rather than attention's cap.
        logits = tf.matmul(tf.reshape(hidden, new_shape=(1, config.hidden_size)), w_head)
        z = logits / LOGIT_SOFTCAP
        return tf.tanh(z) * LOGIT_SOFTCAP

    @lm_head.converter("w_head")
    def _(
        head_weight_raw: ConstTensor[(config.vocab_size, config.hidden_size), _DT],
    ) -> Tensor[(config.hidden_size, config.vocab_size), _DT]:
        # HF stores the head as (vocab, hidden); the matmul above wants it the
        # other way. Gemma-2 ties its head, so this input is the embedding table.
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
