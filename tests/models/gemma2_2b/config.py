"""Shared fixtures for the Gemma-2-2B single decoder-layer HIR description.

The fourth of the Phase 0 dense/near-dense "打样" models (Qwen3-1.7B,
Qwen2.5-1.5B, MiniCPM3, Gemma-2), authored against the same three-file
template as ``tests/models/qwen3_1_7b/`` (``config.py`` + ``model/decoder_layer.py``
+ ``test_decoder_layer.py``), run on macOS with no CUDA: **cpu + f32 only**.

Pins the value oracle and the model contract every component test in this
package shares — the Hugging Face reference (``transformers``
``Gemma2DecoderLayer`` built from a ``Gemma2Config`` at the Gemma-2-2B
dimensions, random weights at a fixed seed), the model dimensions (GQA 8
query / 4 key-value heads; dense gelu_pytorch_tanh-gated MLP), and the static
single-shot-prefill contract (a fixed ``REAL.s_cap``-token sequence, no KV cache,
no dynamic ``DimVar``).

Gemma-2 has no per-head q_norm/k_norm (that is Qwen3-specific), but adds
three things Qwen3 does not have — all confirmed live against this repo's
``transformers`` install (see module docstrings in ``model/decoder_layer.py``
for how each is composed into HIR):

- ``Gemma2RMSNorm`` computes ``normed * (1.0 + weight)``, not
  ``normed * weight`` (``tf.rms_norm``'s semantics). :func:`rms_gamma` below
  does the ``1.0 + weight`` test-side preprocessing for all four norms in
  this package — the kernel itself never changes.
- attention scaling is ``query_pre_attn_scalar**-0.5`` (0.0625 @ 256), not
  ``head_dim**-0.5``.
- attention logits are soft-capped: ``attn_logit_softcapping *
  tanh(scores / attn_logit_softcapping)`` before the mask is added.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Gemma-2-2B dimensions ────────────────────────────────────────────────
# Public Gemma-2-2B config == `transformers.Gemma2Config()` defaults
# (confirmed live in a REPL against this repo's transformers==5.14.1):
# hidden_size=2304, num_attention_heads=8, num_key_value_heads=4,
# head_dim=256 (explicit — NOT hidden_size // num_attention_heads == 288, so
# unlike qwen3_1_7b, REAL.q_proj != REAL.hidden here), intermediate_size=9216,
# rms_norm_eps=1e-6, query_pre_attn_scalar=256, attn_logit_softcapping=50.0,
# sliding_window=4096, hidden_activation="gelu_pytorch_tanh",
# max_position_embeddings=8192. rope_parameters resolves (rope_parameters=None
# default) to {"rope_theta": 10000.0, "rope_type": "default"} — plain RoPE,
# attention_scaling==1.0, same as qwen3_1_7b's cos/sin cache convention.
@dataclass(frozen=True)
class Gemma2Shape:
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
    query_pre_attn_scalar: int
    attn_softcap: float
    sliding_window: int
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
    s_cap=4,
    dt='f32',
)

# ── Component -> HF submodule map ───────────────────────────────────────
# `self_attention` / `mlp` are pure blocks here (no fused norm, unlike
# qwen3_1_7b) — see the `model/decoder_layer.py` docstring for why Gemma-2's
# four-norm layout makes that the natural fusion boundary. Only
# `decoder_layer` composes the full HF submodule tree.
COMPONENT_HF_SUBMODULES = {
    "input_rms_norm": ("input_layernorm",),
    "self_attention": ("input_layernorm", "self_attn"),
    "mlp": ("mlp",),
    "decoder_layer": (".",),
}


def build_hf_config():
    """Build a ``Gemma2Config`` at the Gemma-2-2B dimensions (one decoder layer)."""
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
        num_hidden_layers=1,
        vocab_size=REAL.vocab,
        max_position_embeddings=REAL.max_pos,
    )


def build_hf_layer(seed=0, device="cpu", dtype=None):
    """Build a ``Gemma2DecoderLayer`` with random weights at a fixed seed.

    ``device`` defaults to ``"cpu"`` (no CUDA on this box — every caller in
    this package either omits ``device`` or passes ``"cpu"`` explicitly).
    """
    import torch  # noqa: PLC0415
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer  # noqa: PLC0415

    cfg = build_hf_config()
    torch.manual_seed(seed)
    layer = Gemma2DecoderLayer(cfg, layer_idx=0).to(device).eval()
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
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding  # noqa: PLC0415

    rotary = Gemma2RotaryEmbedding(cfg).to(device)
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

    No KV cache in this package: every token attends only within the same
    ``REAL.s_cap`` tile, i.e. ``cur_pos`` is always 0 (mirrors
    ``qwen3_1_7b.config.causal_mask``).
    """
    import torch  # noqa: PLC0415

    q_pos = torch.arange(seq, device=device).unsqueeze(1)
    k_pos = torch.arange(seq, device=device).unsqueeze(0)
    mask = torch.where(k_pos <= q_pos, 0.0, float("-inf"))
    if dtype is not None:
        mask = mask.to(dtype)
    return mask.view(1, 1, seq, seq)


def sliding_causal_mask(seq, window, device="cpu", dtype=None):
    """Additive causal **+ sliding-window** mask ``[1, 1, seq, seq]``: same as
    :func:`causal_mask`, but also ``-inf`` where ``j < q_pos - window`` (a key
    more than ``window`` tokens in the past).

    Optional coverage for the sliding-window gotcha (hint #6): at this
    package's ``REAL.s_cap``, ``REAL.sliding_window`` (4096) never actually binds, so
    the four required component tests all use the plain :func:`causal_mask`.
    This helper builds a deliberately small ``window`` so
    ``test_self_attention_sliding_window_evaluate`` exercises a mask that
    truly differs from causal-only, proving the HIR composition isn't
    secretly assuming "causal == the whole mask story".
    """
    import torch  # noqa: PLC0415

    q_pos = torch.arange(seq, device=device).unsqueeze(1)
    k_pos = torch.arange(seq, device=device).unsqueeze(0)
    allowed = (k_pos <= q_pos) & (k_pos >= (q_pos - window))
    mask = torch.where(allowed, 0.0, float("-inf"))
    if dtype is not None:
        mask = mask.to(dtype)
    return mask.view(1, 1, seq, seq)


def linear_weight(linear):
    """HF ``nn.Linear.weight`` ``[out, in]`` -> kernel matmul layout
    ``[1, in, out]`` (the kernel convention is ``x[1,S,in] @ w[1,in,out]``,
    the transpose/pack "weight preprocessing" the task calls for happening in
    test code, not in the kernel)."""
    return linear.weight.t().unsqueeze(0).contiguous()


def rms_gamma(hf_norm):
    """``Gemma2RMSNorm.forward`` computes ``normed * (1.0 + weight)``;
    ``tf.rms_norm`` computes ``normed * weight``. Pre-add the ``1.0`` here
    (test-side preprocessing, not a kernel/op change) so the kernel's plain
    ``* weight`` semantics reproduce HF's ``(1 + weight)`` scaling. Applies to
    all four norms in this package (input / post-attention /
    pre-feedforward / post-feedforward)."""
    return 1.0 + hf_norm.weight
