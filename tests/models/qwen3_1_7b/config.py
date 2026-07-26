"""Qwen3-1.7B dimensions and the Hugging Face oracle every test in this
package compares against.

Phase 0 "打样": the first of four planned dense/near-dense models (Qwen3-1.7B,
Qwen2.5-1.5B, MiniCPM3, Gemma-2) authored against this same three-file
template (``config.py`` + ``model/decoder_layer.py`` + ``test_decoder_layer.py``,
mirroring ``tests/models/qwen3_5_30b_a3b/``), run on macOS with no CUDA:
**cpu + f32 only**.

Pins the value oracle and the model contract every component test in this
package shares:

- the Hugging Face reference (``transformers`` ``Qwen3DecoderLayer`` built
  from a ``Qwen3Config`` at (approximately) the Qwen3-1.7B dimensions, random
  weights at a fixed seed),
- the model dimensions (GQA 16 query / 8 key-value heads; a dense SwiGLU MLP
  — no MoE router/gather, unlike the ``qwen3_5_30b_a3b`` sibling package),
- the static single-shot-prefill contract (a fixed ``S_CAP``-token sequence,
  no KV cache, no dynamic ``DimVar`` — Phase 0 validates op-composition
  correctness, not context-length scaling), and
- the component -> HF-submodule map.

Component HIR ``@func``s live in ``model/decoder_layer.py``, over this module's
``REAL`` shape; ``decoder_layer.py`` binds the two together. This module holds
only the shape, the HF layer / rope-cache / causal-mask builders, and the
weight-layout helper.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Qwen3-1.7B dimensions ────────────────────────────────────────────────
# Public Qwen3-1.7B config.json values (https://huggingface.co/Qwen/Qwen3-1.7B):
# hidden_size=2048, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
# intermediate_size=6144, rms_norm_eps=1e-6, rope_theta=1e6,
# max_position_embeddings=32768. Fields the model does not pin (attention_bias)
# fall back to the ``Qwen3Config`` default, which is also ``False``.


@dataclass(frozen=True)
class Qwen3Shape:
    """One decoder layer's shape, plus the sequence length and dtype every
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
    s_cap: int
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


# ``s_cap`` is static: no KV cache, no dynamic dims -- a single-shot prefill
# oracle (``cur_pos`` is always 0). Op semantics are length-agnostic, but atom
# matching is not: an atom's row granularity has to divide the op's row count,
# and at 4 every matmul here fails that (``4 % 16 != 0`` against the AMX outer
# product) and lists no candidate. 64 is a length that keeps the numerical
# oracle cheap and still divides both modelled granularities.
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
    max_pos=32768,
    s_cap=64,
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


def build_hf_config(shape: Qwen3Shape = REAL):
    """Build a ``Qwen3Config`` at *shape*'s dimensions (one decoder layer)."""
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
        num_hidden_layers=1,
        vocab_size=shape.vocab,
        max_position_embeddings=shape.max_pos,
    )


def build_hf_layer(seed=0, device="cpu", dtype=None):
    """Build a ``Qwen3DecoderLayer`` with random weights at a fixed seed.

    ``device`` defaults to ``"cpu"`` (no CUDA on this box — every caller in
    this package either omits ``device`` or passes ``"cpu"`` explicitly).
    """
    import torch  # noqa: PLC0415
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer  # noqa: PLC0415

    cfg = build_hf_config()
    torch.manual_seed(seed)
    layer = Qwen3DecoderLayer(cfg, layer_idx=0).to(device).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0.0, 0.05)
    if dtype is not None:
        layer = layer.to(dtype)
    return layer


def rope_caches(cfg, max_pos, device="cpu", dtype=None):
    """Full cos / sin caches ``[max_pos, head_dim]`` from the HF rotary embedding.

    Row ``p`` is the rotary embedding for absolute position ``p``, so gathering
    by ``pos_ids`` reproduces the cos / sin the HF attention applies.
    """
    import torch  # noqa: PLC0415
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding  # noqa: PLC0415

    rotary = Qwen3RotaryEmbedding(cfg).to(device)
    position_ids = torch.arange(max_pos, device=device).unsqueeze(0)
    ref = torch.zeros(1, max_pos, cfg.hidden_size, device=device)
    cos, sin = rotary(ref, position_ids)
    cos, sin = cos[0], sin[0]
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def causal_mask(seq, device="cpu", dtype=None):
    """Additive causal mask ``[1, 1, seq, seq]``: 0 where query ``i`` may
    attend key ``j`` (``j <= i``), ``-inf`` otherwise.

    Unlike the ``qwen3_5_30b_a3b`` sibling's ``additive_causal_mask`` /
    ``decode_attn_mask`` (which both take a ``cur_pos`` — prior KV-cache
    length), this package has no KV cache: every token attends only within
    the same ``S_CAP`` tile, i.e. ``cur_pos`` is always 0.
    """
    import torch  # noqa: PLC0415

    q_pos = torch.arange(seq, device=device).unsqueeze(1)
    k_pos = torch.arange(seq, device=device).unsqueeze(0)
    mask = torch.where(k_pos <= q_pos, 0.0, float("-inf"))
    if dtype is not None:
        mask = mask.to(dtype)
    return mask.view(1, 1, seq, seq)


def linear_weight(linear):
    """HF ``nn.Linear.weight`` ``[out, in]`` -> kernel matmul layout
    ``[1, in, out]`` (the kernel convention is ``x[1,S,in] @ w[1,in,out]``,
    the transpose/pack "weight preprocessing" the task calls for happening in
    test code, not in the kernel)."""
    return linear.weight.t().unsqueeze(0).contiguous()
