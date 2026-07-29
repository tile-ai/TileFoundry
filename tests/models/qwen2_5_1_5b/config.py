"""Shared fixtures for the Qwen2.5-1.5B dense decoder-layer HIR description.

Phase 0 "打样": second of four planned dense/near-dense models (Qwen3-1.7B,
Qwen2.5-1.5B, MiniCPM3, Gemma-2) authored against the same three-file
template (``config.py`` + ``model.py`` + ``test_decoder_layer.py``,
mirroring ``tests/models/qwen3_1_7b/``, itself mirroring
the layout every model package here shares), run on macOS with no CUDA: **cpu + f32
only**.

Pins the value oracle and the model contract every component test in this
package shares:

- the Hugging Face reference (``transformers`` ``Qwen2DecoderLayer`` built
  from a ``Qwen2Config`` at (approximately) the Qwen2.5-1.5B dimensions,
  random weights at a fixed seed),
- the model dimensions (GQA 12 query / 2 key-value heads; a dense SwiGLU MLP
  — no MoE router/gather),
- the decode contract: one token per step (``seq_len`` is 1), the active context
  length ``ctx_len`` as the single dynamic dimension, and the KV cache passed as
  explicit tensors in and out — Hugging Face's ``past_key_values`` is never
  constructed, on either side of the comparison, and
- the component -> HF-submodule map.

Component HIR ``@func``s live in ``model.py`` (the ``@module
class`` authoring style, per ``qwen3_1_7b/model.py``); this
module only holds the shared dims, the HF layer / rope-cache / causal-mask
builders, and the weight-layout helper, so every test file composes them
rather than duplicating the description.

Two structural differences from the ``qwen3_1_7b`` sibling (see
``model.py`` docstring for the HIR-level detail):

- Qwen2 attention has no per-head ``q_norm`` / ``k_norm``: HF
  ``Qwen2Attention`` applies RoPE directly to the raw ``q_proj`` /
  ``k_proj`` output, no intervening RMSNorm.
- Qwen2's ``q_proj`` / ``k_proj`` / ``v_proj`` each carry a bias (``o_proj``
  does not) — hardcoded in HF ``Qwen2Attention.__init__`` (``bias=True`` /
  ``True`` / ``True`` / ``False`` respectively), not gated by a
  ``Qwen2Config`` flag at all (unlike Qwen3's ``attention_bias`` field,
  which ``Qwen2Config`` simply has no equivalent of).
"""
from __future__ import annotations

from dataclasses import dataclass

from tests.models import decode_oracle as oracle

# ── Qwen2.5-1.5B dimensions ──────────────────────────────────────────────
# Every dimension below comes from the published configuration, pinned so the claim
# is checkable rather than quoted. It had been quoted, and `max_position_embeddings`
# was written here as 32768 -- the `Qwen2Config` dataclass default -- where the
# published file says 131072. Agreeing with a library default is not the same fact
# as agreeing with the model.
#
#: Where the dimensions come from.
SOURCE_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json"
#: The commit the values were read at.
SOURCE_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
#: sha256 of that file as fetched.
SOURCE_SHA256 = "0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77"
#
# hidden_size=1536, num_attention_heads=12, num_key_value_heads=2, head_dim=128,
# intermediate_size=8960, rms_norm_eps=1e-6, rope_theta=1e6, vocab_size=151936,
# num_hidden_layers=28, max_position_embeddings=131072.
@dataclass(frozen=True)
class Qwen25Shape:
    """One decoder layer's shape, plus the context envelope and dtype every
    kernel in this package is authored at."""

    hidden: int
    head_dim: int
    n_q_heads: int
    n_kv_heads: int
    intermediate: int
    rms_eps: float
    rope_theta: float
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
# reads. ``max_ctx`` matches ``max_pos``: a position beyond the rotary cache has
# no embedding to gather, so a longer context is unrepresentable rather than slow.
SEQ_LEN = 1

REAL = Qwen25Shape(
    hidden=1536,
    head_dim=128,
    n_q_heads=12,
    n_kv_heads=2,
    intermediate=8960,
    rms_eps=1e-06,
    rope_theta=1000000.0,
    vocab=151936,
    max_pos=131072,
    max_ctx=131072,
    n_layers=28,
    dt='f32',
)

# ── Component -> HF submodule map ───────────────────────────────────────
# Each component's HIR is validated against these submodules of a single
# ``Qwen2DecoderLayer``. ``self_attention`` and ``mlp`` each fuse their
# preceding RMSNorm (see ``model.py`` docstring), so their HF
# comparison composes the norm + block rather than the block alone.
COMPONENT_HF_SUBMODULES = {
    "input_rms_norm": ("input_layernorm",),
    "self_attention": ("input_layernorm", "self_attn"),
    "mlp": ("post_attention_layernorm", "mlp"),
    "decoder_layer": (".",),
}


def build_hf_config(*, layers: int = 1):
    """Build a ``Qwen2Config`` at the Qwen2.5-1.5B dimensions.

    ``layers`` defaults to one, which is what a component test needs. The
    complete-decoder reference asks for ``REAL.n_layers`` instead.
    """
    from transformers import Qwen2Config  # noqa: PLC0415

    return Qwen2Config(
        hidden_size=REAL.hidden,
        head_dim=REAL.head_dim,
        num_attention_heads=REAL.n_q_heads,
        num_key_value_heads=REAL.n_kv_heads,
        intermediate_size=REAL.intermediate,
        rms_norm_eps=REAL.rms_eps,
        rope_theta=REAL.rope_theta,
        num_hidden_layers=layers,
        vocab_size=REAL.vocab,
        max_position_embeddings=REAL.max_pos,
    )


def build_hf_layer(seed=0, device="cpu", dtype=None):
    """Build a ``Qwen2DecoderLayer`` with random weights at a fixed seed.

    ``device`` defaults to ``"cpu"`` (no CUDA on this box — every caller in
    this package either omits ``device`` or passes ``"cpu"`` explicitly).
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer  # noqa: PLC0415

    return oracle.randomised(
        lambda: Qwen2DecoderLayer(build_hf_config(), layer_idx=0), seed, device, dtype
    )


def rope_caches(cfg, max_pos, device="cpu", dtype=None):
    """Full cos / sin caches ``[max_pos, head_dim]`` from the HF rotary embedding.

    Row ``p`` is the rotary embedding for absolute position ``p``, so gathering
    by ``pos_ids`` reproduces the cos / sin the HF attention applies.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding  # noqa: PLC0415

    return oracle.rope_caches(Qwen2RotaryEmbedding, cfg, max_pos, device, dtype)


def _key_value_of(layer, normed):
    """*layer*'s pre-rotary key and its value, head-major.

    The one step of the oracle that is Qwen2's own, and it differs from Qwen3 in
    both directions: there is no per-head norm on the key, and the projections
    carry a bias -- which needs no handling here only because it is inside the
    ``nn.Linear`` this calls.
    """
    attention = layer.self_attn
    heads = (1, normed.shape[1], REAL.n_kv_heads, REAL.head_dim)
    key = attention.k_proj(normed).view(heads).transpose(1, 2)
    value = attention.v_proj(normed).view(heads).transpose(1, 2)
    return key, value


def _apply_rotary():
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb  # noqa: PLC0415

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


def build_hf_decoder(seed=0, device="cpu", dtype=None):
    """The complete ``REAL.n_layers``-layer decoder stack, random at a fixed seed.

    A ``Qwen2ForCausalLM`` rather than the base model: the decoder's own boundary
    is still hidden states in and hidden states out, but the root's weights include
    the head, and the head exists only on the causal LM. Its layers and final norm
    are reached through ``.model``.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM  # noqa: PLC0415

    return oracle.randomised(
        lambda: Qwen2ForCausalLM(build_hf_config(layers=REAL.n_layers)), seed, device, dtype
    )


def decoder_context_kv(model, hidden_ctx, device="cpu"):
    """Per-layer ``(k_cache, v_cache)`` for *hidden_ctx*, in layer order."""
    cos, sin = rope_caches(build_hf_config(), hidden_ctx.shape[1], device=device)
    return oracle.stack_context_kv(
        model.model.layers, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
    )


def decoder_decode_reference(model, hidden_ctx, hidden_new):
    """The decoder stack's output for *hidden_new* decoded after *hidden_ctx*."""
    device = hidden_ctx.device.type
    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    return oracle.decode_reference(
        model.model.layers, hidden_ctx, hidden_new, cos, sin, final_norm=model.model.norm
    )


def linear_weight(linear):
    """HF ``nn.Linear.weight`` ``[out, in]`` -> kernel matmul layout
    ``[1, in, out]`` (the kernel convention is ``x[1,S,in] @ w[1,in,out]``,
    the transpose/pack "weight preprocessing" the task calls for happening in
    test code, not in the kernel). ``nn.Linear.bias`` (when present) needs no
    such transpose — it is used as-is."""
    return linear.weight.t().unsqueeze(0).contiguous()
