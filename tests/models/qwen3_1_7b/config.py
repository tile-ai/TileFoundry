"""Qwen3-1.7B dimensions and the Hugging Face oracle every test in this
package compares against.

Phase 0 "打样": the first of four planned dense/near-dense models (Qwen3-1.7B,
Qwen2.5-1.5B, MiniCPM3, Gemma-2) authored against this same three-file
template (``config.py`` + ``model/decoder_layer.py`` + ``test_decoder_layer.py``,
mirroring the layout every model package here shares), run on macOS with no
CUDA:
**cpu + f32 only**.

Pins the value oracle and the model contract every component test in this
package shares:

- the Hugging Face reference (``transformers`` ``Qwen3DecoderLayer`` built
  from a ``Qwen3Config`` at (approximately) the Qwen3-1.7B dimensions, random
  weights at a fixed seed),
- the model dimensions (GQA 16 query / 8 key-value heads; a dense SwiGLU MLP
  — no MoE router/gather, unlike the ``qwen3_5_35b_a3b`` sibling package),
- the decode contract: one token per step (``seq_len`` is 1), the active
  context length ``ctx_len`` as the single dynamic dimension, and the KV cache
  passed as explicit tensors in and out — Hugging Face's ``past_key_values`` is
  never constructed, on either side of the comparison, and
- the component -> HF-submodule map.

Component HIR ``@func``s live in ``model/decoder_layer.py``, over this module's
``REAL`` shape; ``decoder_layer.py`` binds the two together. This module holds
only the shape, the HF layer / rope-cache / causal-mask builders, and the
weight-layout helper.
"""
from __future__ import annotations

from dataclasses import dataclass

from tests.models import decode_oracle as oracle

# ── Qwen3-1.7B dimensions ────────────────────────────────────────────────
# Every dimension below comes from the published configuration, pinned so the claim
# is checkable rather than quoted. It had been quoted, and one of the numbers was
# wrong: `max_position_embeddings` was written here as 32768 where the published
# file says 40960, which nothing could have caught while the source was named only
# by repository.
#
#: Where the dimensions come from.
SOURCE_URL = "https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json"
#: The commit the values were read at. A full sha rather than a branch, because a
#: branch names whatever it points at today and that is the thing being pinned
#: against.
SOURCE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
#: sha256 of that file as fetched.
SOURCE_SHA256 = "1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
#
# hidden_size=2048, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
# intermediate_size=6144, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=151936,
# num_hidden_layers=28, max_position_embeddings=40960. Fields the model does not
# pin (attention_bias) fall back to the ``Qwen3Config`` default, which is also
# ``False``.


@dataclass(frozen=True)
class Qwen3Shape:
    """One decoder layer's shape, plus the context envelope and dtype every
    kernel in this package is authored at."""

    hidden: int
    head_dim: int
    n_q_heads: int
    n_kv_heads: int
    intermediate: int
    rms_eps: float
    rope_theta: float
    attention_bias: bool
    vocab: int
    max_pos: int
    max_ctx: int
    n_layers: int
    dt: str

    @property
    def gqa_group(self) -> int:
        """Query heads sharing one key/value head."""
        return self.n_q_heads // self.n_kv_heads

    @property
    def q_proj(self) -> int:
        return self.n_q_heads * self.head_dim

    @property
    def kv_proj(self) -> int:
        return self.n_kv_heads * self.head_dim


# One token per step, so the only dynamic dimension is the context the step
# reads. ``max_ctx`` is the largest context the kernels are authored for; it
# matches ``max_pos`` because a position beyond the rotary cache has no
# embedding to gather, so a longer context would be unrepresentable rather than
# merely slow.
SEQ_LEN = 1

REAL = Qwen3Shape(
    hidden=2048,
    head_dim=128,
    n_q_heads=16,
    n_kv_heads=8,
    intermediate=6144,
    rms_eps=1e-6,
    rope_theta=1_000_000.0,
    attention_bias=False,
    vocab=151936,
    max_pos=40960,
    max_ctx=40960,
    n_layers=28,
    dt="f32",
)

# ── Component -> HF submodule map ───────────────────────────────────────
# Each component's HIR is validated against these submodules of a single
# ``Qwen3DecoderLayer``. ``self_attention`` and ``mlp`` each fuse their
# preceding RMSNorm (see ``model/decoder_layer.py`` docstring), so their HF
# comparison composes the norm + block rather than the block alone.
COMPONENT_HF_SUBMODULES = {
    "input_rms_norm": ("input_layernorm",),
    "self_attention": ("input_layernorm", "self_attn"),
    "mlp": ("post_attention_layernorm", "mlp"),
    "decoder_layer": (".",),
}


def build_hf_config(shape: Qwen3Shape = REAL, *, layers: int = 1):
    """Build a ``Qwen3Config`` at *shape*'s dimensions.

    ``layers`` defaults to one, which is what a component test needs: a single
    layer's submodules, with none of the cost of instantiating the rest. The
    complete-model reference asks for ``shape.n_layers`` instead.
    """
    from transformers import Qwen3Config  # noqa: PLC0415

    return Qwen3Config(
        hidden_size=shape.hidden,
        head_dim=shape.head_dim,
        num_attention_heads=shape.n_q_heads,
        num_key_value_heads=shape.n_kv_heads,
        intermediate_size=shape.intermediate,
        rms_norm_eps=shape.rms_eps,
        rope_theta=shape.rope_theta,
        attention_bias=shape.attention_bias,
        num_hidden_layers=layers,
        vocab_size=shape.vocab,
        max_position_embeddings=shape.max_pos,
    )


def build_hf_layer(seed=0, device="cpu", dtype=None):
    """Build a ``Qwen3DecoderLayer`` with random weights at a fixed seed.

    ``device`` defaults to ``"cpu"`` (no CUDA on this box — every caller in
    this package either omits ``device`` or passes ``"cpu"`` explicitly).
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer  # noqa: PLC0415

    return oracle.randomised(
        lambda: Qwen3DecoderLayer(build_hf_config(), layer_idx=0), seed, device, dtype
    )


def rope_caches(cfg, max_pos, device="cpu", dtype=None):
    """Full cos / sin caches ``[max_pos, head_dim]`` from the HF rotary embedding.

    Row ``p`` is the rotary embedding for absolute position ``p``, so gathering
    by ``pos_ids`` reproduces the cos / sin the HF attention applies.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding  # noqa: PLC0415

    return oracle.rope_caches(Qwen3RotaryEmbedding, cfg, max_pos, device, dtype)


def _key_value_of(layer, normed):
    """*layer*'s pre-rotary key and its value, head-major.

    The one step of the oracle that is Qwen3's own: the key is normalised per
    head, and no projection carries a bias.
    """
    attention = layer.self_attn
    heads = (1, normed.shape[1], REAL.n_kv_heads, REAL.head_dim)
    key = attention.k_norm(attention.k_proj(normed).view(heads)).transpose(1, 2)
    value = attention.v_proj(normed).view(heads).transpose(1, 2)
    return key, value


def _apply_rotary():
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # noqa: PLC0415

    return apply_rotary_pos_emb


def context_kv(layer, hidden_ctx, device="cpu"):
    """The KV cache *layer* would hold for *hidden_ctx*, as explicit tensors."""
    cos, sin = rope_caches(build_hf_config(), hidden_ctx.shape[1], device=device)
    return oracle.context_kv(
        layer, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
    )


def decode_reference(layer, hidden_ctx, hidden_new, device="cpu"):
    """Hugging Face's output for *hidden_new* decoded after *hidden_ctx*."""
    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    return oracle.decode_reference([layer], hidden_ctx, hidden_new, cos, sin)


def build_hf_decoder(seed=0, device="cpu", dtype=None, shape: Qwen3Shape = REAL):
    """The complete ``shape.n_layers``-layer decoder stack, random at a fixed seed.

    A ``Qwen3Model`` is built for its layers and its final norm; its token
    embedding is not part of what this returns, because the decoder's boundary is
    hidden states in and hidden states out. Stacking one layer's verified
    behaviour is not the same as the stack behaving, which is why this exists
    separately from ``build_hf_layer``: layer order, the final norm, and the
    residual thread between layers are only observable here.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model  # noqa: PLC0415

    return oracle.randomised(
        lambda: Qwen3Model(build_hf_config(shape, layers=shape.n_layers)),
        seed, device, dtype,
    )


def decoder_context_kv(model, hidden_ctx, device="cpu"):
    """Per-layer ``(k_cache, v_cache)`` for *hidden_ctx*, in layer order."""
    cos, sin = rope_caches(build_hf_config(), hidden_ctx.shape[1], device=device)
    return oracle.stack_context_kv(
        model.layers, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
    )


def decoder_decode_reference(model, hidden_ctx, hidden_new):
    """The decoder stack's output for *hidden_new* decoded after *hidden_ctx*."""
    device = hidden_ctx.device.type
    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    return oracle.decode_reference(
        model.layers, hidden_ctx, hidden_new, cos, sin, final_norm=model.norm
    )


def linear_weight(linear):
    """HF ``nn.Linear.weight`` ``[out, in]`` -> kernel matmul layout
    ``[1, in, out]`` (the kernel convention is ``x[1,S,in] @ w[1,in,out]``,
    the transpose/pack "weight preprocessing" the task calls for happening in
    test code, not in the kernel)."""
    return linear.weight.t().unsqueeze(0).contiguous()
