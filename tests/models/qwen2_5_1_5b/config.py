"""Shared fixtures for the Qwen2.5-1.5B dense decoder-layer HIR description.

Phase 0 "打样": second of four planned dense/near-dense models (Qwen3-1.7B,
Qwen2.5-1.5B, MiniCPM3, Gemma-2) authored against the same three-file
template (``config.py`` + ``model/decoder_layer.py`` + ``test_decoder_layer.py``,
mirroring ``tests/models/qwen3_1_7b/``, itself mirroring
``tests/models/qwen3_5_30b_a3b/``), run on macOS with no CUDA: **cpu + f32
only**.

Pins the value oracle and the model contract every component test in this
package shares:

- the Hugging Face reference (``transformers`` ``Qwen2DecoderLayer`` built
  from a ``Qwen2Config`` at (approximately) the Qwen2.5-1.5B dimensions,
  random weights at a fixed seed),
- the model dimensions (GQA 12 query / 2 key-value heads; a dense SwiGLU MLP
  — no MoE router/gather),
- the static single-shot-prefill contract (a fixed ``REAL.s_cap``-token sequence,
  no KV cache, no dynamic ``DimVar`` — Phase 0 validates op-composition
  correctness, not context-length scaling), and
- the component -> HF-submodule map.

Component HIR ``@func``s live in ``model/decoder_layer.py`` (the ``@module
class`` authoring style, per ``qwen3_1_7b/model/decoder_layer.py``); this
module only holds the shared dims, the HF layer / rope-cache / causal-mask
builders, and the weight-layout helper, so every test file composes them
rather than duplicating the description.

Two structural differences from the ``qwen3_1_7b`` sibling (see
``model/decoder_layer.py`` docstring for the HIR-level detail):

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


# ── Qwen2.5-1.5B dimensions ──────────────────────────────────────────────
# Public Qwen2.5-1.5B config.json values (https://huggingface.co/Qwen/Qwen2.5-1.5B):
# hidden_size=1536, num_attention_heads=12, num_key_value_heads=2, head_dim=128,
# intermediate_size=8960, rms_norm_eps=1e-6, rope_theta=1e6,
# max_position_embeddings=32768. Everything below matches those exactly;
# ``REAL.vocab`` / ``REAL.max_pos`` already equal the ``Qwen2Config`` dataclass
# defaults (confirmed against the installed ``transformers`` package) and
# are kept explicit only for parity with the ``qwen3_1_7b`` template.
@dataclass(frozen=True)
class Qwen25Shape:
    """One decoder layer's shape, plus the sequence length and dtype every
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


REAL = Qwen25Shape(
    hidden=1536,
    head_dim=128,
    n_q_heads=12,
    n_kv_heads=2,
    intermediate=8960,
    rms_eps=1e-06,
    rope_theta=1000000.0,
    vocab=151936,
    max_pos=32768,
    s_cap=4,
    dt='f32',
)

# ── Component -> HF submodule map ───────────────────────────────────────
# Each component's HIR is validated against these submodules of a single
# ``Qwen2DecoderLayer``. ``self_attention`` and ``mlp`` each fuse their
# preceding RMSNorm (see ``model/decoder_layer.py`` docstring), so their HF
# comparison composes the norm + block rather than the block alone.
COMPONENT_HF_SUBMODULES = {
    "input_rms_norm": ("input_layernorm",),
    "self_attention": ("input_layernorm", "self_attn"),
    "mlp": ("post_attention_layernorm", "mlp"),
    "decoder_layer": (".",),
}


def build_hf_config():
    """Build a ``Qwen2Config`` at the Qwen2.5-1.5B dimensions (one decoder layer)."""
    from transformers import Qwen2Config  # noqa: PLC0415

    return Qwen2Config(
        hidden_size=REAL.hidden,
        head_dim=REAL.head_dim,
        num_attention_heads=REAL.n_q_heads,
        num_key_value_heads=REAL.n_kv_heads,
        intermediate_size=REAL.intermediate,
        rms_norm_eps=REAL.rms_eps,
        rope_theta=REAL.rope_theta,
        num_hidden_layers=1,
        vocab_size=REAL.vocab,
        max_position_embeddings=REAL.max_pos,
    )


def build_hf_layer(seed=0, device="cpu", dtype=None):
    """Build a ``Qwen2DecoderLayer`` with random weights at a fixed seed.

    ``device`` defaults to ``"cpu"`` (no CUDA on this box — every caller in
    this package either omits ``device`` or passes ``"cpu"`` explicitly).
    """
    import torch  # noqa: PLC0415
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer  # noqa: PLC0415

    cfg = build_hf_config()
    torch.manual_seed(seed)
    layer = Qwen2DecoderLayer(cfg, layer_idx=0).to(device).eval()
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
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding  # noqa: PLC0415

    rotary = Qwen2RotaryEmbedding(cfg).to(device)
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

    There is no KV cache in this package: every token attends only within the
    same ``REAL.s_cap`` tile, i.e. ``cur_pos`` is always 0.
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
    test code, not in the kernel). ``nn.Linear.bias`` (when present) needs no
    such transpose — it is used as-is."""
    return linear.weight.t().unsqueeze(0).contiguous()
