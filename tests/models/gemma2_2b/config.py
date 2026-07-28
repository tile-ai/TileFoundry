"""Shared fixtures for the Gemma-2-2B decoder HIR description.

Pins the value oracle and the model contract every test in this package shares:

- the Hugging Face reference (``transformers`` ``Gemma2DecoderLayer`` /
  ``Gemma2Model`` built from a ``Gemma2Config`` at the Gemma-2-2B dimensions,
  random weights at a fixed seed),
- the model dimensions (GQA 8 query / 4 key-value heads, ``head_dim`` 256 stated
  explicitly rather than derived; a dense ``gelu_pytorch_tanh``-gated MLP),
- the decode contract: one token per step (``SEQ_LEN`` is 1), the active context
  length ``ctx_len`` as the single dynamic dimension, and the KV cache passed as
  explicit tensors in and out -- Hugging Face's ``past_key_values`` is never
  constructed, on either side of the comparison, and
- the component -> HF-submodule map.

Everything that builds the oracle is delegated to ``tests.models.decode_oracle``,
which states the construction once for the whole corpus. Two things are Gemma-2's
own and stay here: :func:`_key_value_of`, how its attention turns normed hidden
states into a key and a value, and which ``apply_rotary_pos_emb`` rotates them.

Four things Gemma-2 does that the Qwen siblings do not, all confirmed against
the installed ``transformers`` (5.14.1) rather than taken from documentation:

- ``Gemma2RMSNorm.forward`` computes ``normed * (1.0 + weight)``, not
  ``normed * weight`` (``tf.rms_norm``'s semantics). :func:`rms_gamma` does the
  ``1.0 + weight`` preprocessing test-side for all five norms this package
  touches (four per layer plus the stack's final one); the kernel never changes.
- attention scaling is ``query_pre_attn_scalar**-0.5`` (0.0625 at 256), not
  ``head_dim**-0.5`` (which would be 0.0625 as well at ``head_dim`` 256 --
  numerically indistinguishable here, so ``scale`` is read off
  ``self_attn.scaling`` rather than recomputed, and a config where the two
  differ would still be reproduced).
- attention logits are soft-capped: ``attn_logit_softcapping *
  tanh(scores / attn_logit_softcapping)``, applied to the raw scaled scores.
- the MLP activation is ``gelu_pytorch_tanh``, not SwiGLU's ``silu``.

Two traps in the Hugging Face side, both load-bearing:

``attn_implementation`` is pinned to ``"eager"`` below. ``Gemma2Model``'s
``PreTrainedModel.__init__`` otherwise resolves it to ``"sdpa"``, and
``sdpa_attention_forward`` takes no ``softcap`` argument -- it swallows the one
``Gemma2Attention.forward`` passes in ``**kwargs`` and silently drops the
soft-capping. A stack oracle built that way would disagree with a *correct*
kernel. (A standalone ``Gemma2DecoderLayer`` leaves ``_attn_implementation`` at
``None`` and falls back to ``eager_attention_forward``, so the single-layer
oracle was already capped; pinning it makes both sides say so.)

``max_ctx`` is ``sliding_window``, not ``max_position_embeddings``. Gemma-2
alternates layer types -- ``config.layer_types`` is
``["sliding_attention", "full_attention", ...]``, and ``Gemma2Model.forward``
hands each layer a different mask accordingly. The kernels here describe full
attention only, which is exactly right for a context no longer than the window
(a window removes positions from the *front*, so with ``total <= 4096`` nothing
is removed) and wrong beyond it. Rather than let that go quiet,
:func:`_within_window` raises, and the envelope stops where the claim does.
"""
from __future__ import annotations

from dataclasses import dataclass

from tests.models import decode_oracle as oracle

# ── Gemma-2-2B dimensions ────────────────────────────────────────────────
#: Where the dimensions come from.
SOURCE_URL = "https://huggingface.co/google/gemma-2-2b/blob/main/config.json"
#: Why there is no revision and no digest for it, measured rather than assumed.
#:
#: The repository is gated. Fetching the file unauthenticated returns "Access to
#: model google/gemma-2-2b is restricted. You must have access to it and be
#: authenticated to access it. Please log in." -- 125 bytes of prose, not a
#: configuration. So a digest could only be of something else, and a digest of
#: something else is worse than none: it looks like a pin.
#:
#: What stands in for it is stated and checkable without the network: every field
#: below equals the installed `transformers.Gemma2Config()` default, and
#: `test_provenance.py` holds it to that. That is a weaker claim -- it says the
#: dimensions match the library's idea of this model rather than the model's own
#: published file -- and it is written down as weaker rather than presented as a pin.
SOURCE_UNPINNED_REASON = (
    "google/gemma-2-2b is a gated repository: an unauthenticated fetch of "
    "config.json returns an access-restricted message rather than the file, so no "
    "digest of the published configuration can be taken here. The dimensions are "
    "held to the installed transformers Gemma2Config defaults instead."
)
# Public google/gemma-2-2b config.json == `transformers.Gemma2Config()` defaults
# (checked field by field against the installed transformers): hidden_size=2304,
# num_hidden_layers=26, num_attention_heads=8, num_key_value_heads=4,
# head_dim=256 (explicit — NOT hidden_size // num_attention_heads == 288, so
# unlike the Qwen siblings REAL.q_proj != REAL.hidden here),
# intermediate_size=9216, rms_norm_eps=1e-6, query_pre_attn_scalar=256,
# attn_logit_softcapping=50.0, sliding_window=4096,
# hidden_activation="gelu_pytorch_tanh", max_position_embeddings=8192,
# vocab_size=256000. rope_parameters resolves (rope_parameters=None default) to
# {"rope_theta": 10000.0, "rope_type": "default"} — plain RoPE,
# attention_scaling == 1.0, same cos/sin cache convention as the Qwen siblings.
@dataclass(frozen=True)
class Gemma2Shape:
    """One decoder layer's shape, plus the stack's depth, the context envelope
    and the dtype every kernel in this package is authored at."""

    hidden: int
    head_dim: int
    n_q_heads: int
    n_kv_heads: int
    intermediate: int
    rms_eps: float
    rope_theta: float
    attention_bias: bool
    query_pre_attn_scalar: int
    attn_softcap: float
    sliding_window: int
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
# reads.
SEQ_LEN = 1

REAL = Gemma2Shape(
    hidden=2304,
    head_dim=256,
    n_q_heads=8,
    n_kv_heads=4,
    intermediate=9216,
    rms_eps=1e-06,
    rope_theta=10000.0,
    attention_bias=False,
    query_pre_attn_scalar=256,
    attn_softcap=50.0,
    sliding_window=4096,
    vocab=256000,
    max_pos=8192,
    # The window, not the position envelope: past it, half the layers would read
    # a windowed cache these kernels do not describe (see module docstring).
    max_ctx=4096,
    n_layers=26,
    dt='f32',
)

# ── Component -> HF submodule map ───────────────────────────────────────
# `self_attention` / `mlp` are pure blocks here (no fused norm, unlike the Qwen
# siblings) — Gemma-2 sandwiches both with norms on both sides, so fusing either
# would be one-sided. See `model/decoder_layer.py`'s docstring. Their tests
# apply the HF norm in plain torch before calling the kernel, which is why the
# norm does not appear in their entry here.
COMPONENT_HF_SUBMODULES = {
    "input_rms_norm": ("input_layernorm",),
    "self_attention": ("self_attn",),
    "mlp": ("mlp",),
    "decoder_layer": (".",),
}


def _within_window(total: int) -> int:
    """*total* positions, checked against Gemma-2's sliding window.

    Half of Gemma-2's layers attend within ``sliding_window`` rather than over
    the whole context, and these kernels describe full attention. The two agree
    exactly while the window does not bind and not at all once it does, so the
    boundary is enforced rather than documented.
    """
    if total > REAL.sliding_window:
        raise ValueError(
            f"{total} positions exceeds Gemma-2's sliding window "
            f"({REAL.sliding_window}); half the layers would attend a window "
            f"rather than the whole context, which these kernels do not describe"
        )
    return total


def build_hf_config(*, layers: int = 1):
    """Build a ``Gemma2Config`` at the Gemma-2-2B dimensions.

    ``layers`` defaults to one, which is what a component test needs. The
    complete-decoder reference asks for ``REAL.n_layers`` instead.

    ``attn_implementation="eager"`` is not a preference: it is what keeps the
    attention-logit soft-capping in the oracle (see module docstring).
    """
    from transformers import Gemma2Config  # noqa: PLC0415

    return Gemma2Config(
        hidden_size=REAL.hidden,
        head_dim=REAL.head_dim,
        num_attention_heads=REAL.n_q_heads,
        num_key_value_heads=REAL.n_kv_heads,
        intermediate_size=REAL.intermediate,
        rms_norm_eps=REAL.rms_eps,
        attention_bias=REAL.attention_bias,
        query_pre_attn_scalar=REAL.query_pre_attn_scalar,
        attn_logit_softcapping=REAL.attn_softcap,
        sliding_window=REAL.sliding_window,
        hidden_activation="gelu_pytorch_tanh",
        num_hidden_layers=layers,
        vocab_size=REAL.vocab,
        max_position_embeddings=REAL.max_pos,
        attn_implementation="eager",
    )


def build_hf_layer(seed=0, device="cpu", dtype=None):
    """Build a ``Gemma2DecoderLayer`` with random weights at a fixed seed."""
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer  # noqa: PLC0415

    return oracle.randomised(
        lambda: Gemma2DecoderLayer(build_hf_config(), layer_idx=0), seed, device, dtype
    )


def rope_caches(cfg, max_pos, device="cpu", dtype=None):
    """Full cos / sin caches ``[max_pos, head_dim]`` from the HF rotary embedding.

    Row ``p`` is the rotary embedding for absolute position ``p``, so gathering
    by ``pos_ids`` reproduces the cos / sin the HF attention applies.
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding  # noqa: PLC0415

    return oracle.rope_caches(Gemma2RotaryEmbedding, cfg, max_pos, device, dtype)


def _key_value_of(layer, normed):
    """*layer*'s pre-rotary key and its value, head-major.

    The one step of the oracle that is Gemma-2's own, and it is the plainest of
    the corpus: no per-head norm on the key (that is Qwen3's), no projection
    bias (``attention_bias`` is False), just the projection reshaped into heads.
    """
    attention = layer.self_attn
    heads = (1, normed.shape[1], REAL.n_kv_heads, REAL.head_dim)
    key = attention.k_proj(normed).view(heads).transpose(1, 2)
    value = attention.v_proj(normed).view(heads).transpose(1, 2)
    return key, value


def _apply_rotary():
    from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb  # noqa: PLC0415

    return apply_rotary_pos_emb


def context_kv(layer, hidden_ctx, device="cpu"):
    """The KV cache *layer* would hold for *hidden_ctx*, as explicit tensors."""
    total = _within_window(hidden_ctx.shape[1])
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    return oracle.context_kv(
        layer, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
    )


def decode_reference(layer, hidden_ctx, hidden_new, device="cpu"):
    """Hugging Face's output for *hidden_new* decoded after *hidden_ctx*."""
    total = _within_window(hidden_ctx.shape[1] + hidden_new.shape[1])
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    return oracle.decode_reference([layer], hidden_ctx, hidden_new, cos, sin)


def build_hf_decoder(seed=0, device="cpu", dtype=None):
    """The complete ``REAL.n_layers``-layer decoder stack, random at a fixed seed.

    Built for its layers and its final norm; the token embedding is not part of
    what this returns, because the decoder's boundary is hidden states in and
    hidden states out.
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2Model  # noqa: PLC0415

    return oracle.randomised(
        lambda: Gemma2Model(build_hf_config(layers=REAL.n_layers)), seed, device, dtype
    )


def decoder_context_kv(model, hidden_ctx, device="cpu"):
    """Per-layer ``(k_cache, v_cache)`` for *hidden_ctx*, in layer order."""
    total = _within_window(hidden_ctx.shape[1])
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    return oracle.stack_context_kv(
        model.layers, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
    )


def decoder_decode_reference(model, hidden_ctx, hidden_new):
    """The decoder stack's output for *hidden_new* decoded after *hidden_ctx*."""
    device = hidden_ctx.device.type
    total = _within_window(hidden_ctx.shape[1] + hidden_new.shape[1])
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    return oracle.decode_reference(
        model.layers, hidden_ctx, hidden_new, cos, sin, final_norm=model.norm
    )


def linear_weight(linear):
    """HF ``nn.Linear.weight`` ``[out, in]`` -> kernel matmul layout
    ``[1, in, out]`` (the kernel convention is ``x[1,S,in] @ w[1,in,out]``, so
    the transpose is weight preprocessing and belongs in test code, not in the
    kernel)."""
    return oracle.linear_weight(linear)


def rms_gamma(hf_norm):
    """``Gemma2RMSNorm.forward`` computes ``normed * (1.0 + weight)``;
    ``tf.rms_norm`` computes ``normed * weight``. Pre-add the ``1.0`` here
    (test-side preprocessing, not a kernel or op change) so the kernel's plain
    ``* weight`` reproduces HF's ``(1 + weight)`` scaling. Applies to all four
    norms in a layer and to the norm that closes the stack."""
    return 1.0 + hf_norm.weight
