"""Shared fixtures for the Qwen3-1.7B dense decoder-layer HIR description.

Phase 0 "打样": the first of four planned dense/near-dense models (Qwen3-1.7B,
Qwen2.5-1.5B, MiniCPM3, Gemma-2) authored against this same three-file
template (``common.py`` + ``<model>_module.py`` + ``test_<model>_module.py``,
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

Component HIR ``@func``s live in ``qwen3_1_7b_module.py`` (the ``@module
class`` authoring style, per ``qwen3_5_30b_a3b/qwen3_module.py``); this module
only holds the shared dims, the HF layer / rope-cache / causal-mask builders,
and the weight-layout helper, so every test file composes them rather than
duplicating the description.
"""
from __future__ import annotations

# ── Qwen3-1.7B dimensions ────────────────────────────────────────────────
# Public Qwen3-1.7B config.json values (https://huggingface.co/Qwen/Qwen3-1.7B):
# hidden_size=2048, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
# intermediate_size=6144, rms_norm_eps=1e-6, rope_theta=1e6,
# max_position_embeddings=32768. Everything below matches those exactly;
# fields not pinned by the task (e.g. attention_bias) fall back to
# ``Qwen3Config`` defaults (also ``False``, so this is moot here).
HIDDEN = 2048
HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS    # 2 query heads share one kv head
Q_PROJ = NUM_Q_HEADS * HEAD_DIM            # 2048
KV_PROJ = NUM_KV_HEADS * HEAD_DIM          # 1024
INTERMEDIATE = 6144
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
ATTENTION_BIAS = False
VOCAB = 151936
MAX_POS = 32768

# Static: no KV cache, no dynamic dims — a single-shot prefill oracle
# (``cur_pos`` is always 0). Op semantics are length-agnostic, but atom matching
# is not: an atom's row granularity has to divide the op's row count, and at
# S_CAP=4 every matmul here fails that (``4 % 16 != 0`` against the AMX outer
# product) and lists no candidate at all. 64 is a length that keeps the
# numerical oracle cheap and still divides both modelled granularities.
S_CAP = 64

# HIR dtype for every Tensor annotation in this package: f32 everywhere. There
# is no CUDA on this box, so there is no bf16 branch to also cover (contrast
# ``qwen3_5_30b_a3b``, which is bf16-only / GPU-only).
DT = "f32"

# ── Component -> HF submodule map ───────────────────────────────────────
# Each component's HIR is validated against these submodules of a single
# ``Qwen3DecoderLayer``. ``self_attention`` and ``mlp`` each fuse their
# preceding RMSNorm (see ``qwen3_1_7b_module.py`` docstring), so their HF
# comparison composes the norm + block rather than the block alone.
COMPONENT_HF_SUBMODULES = {
    "input_rms_norm": ("input_layernorm",),
    "self_attention": ("input_layernorm", "self_attn"),
    "mlp": ("post_attention_layernorm", "mlp"),
    "decoder_layer": (".",),
}


def build_hf_config():
    """Build a ``Qwen3Config`` at the Qwen3-1.7B dimensions (one decoder layer)."""
    from transformers import Qwen3Config  # noqa: PLC0415

    return Qwen3Config(
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        intermediate_size=INTERMEDIATE,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        attention_bias=ATTENTION_BIAS,
        num_hidden_layers=1,
        vocab_size=VOCAB,
        max_position_embeddings=MAX_POS,
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
