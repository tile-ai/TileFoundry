"""Shared fixtures for the Qwen2.5-1.5B dense decoder-layer HIR description.

Phase 0 "打样": second of four planned dense/near-dense models (Qwen3-1.7B,
Qwen2.5-1.5B, MiniCPM3, Gemma-2) authored against the same three-file
template (``common.py`` + ``<model>_module.py`` + ``test_<model>_module.py``,
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
- the static single-shot-prefill contract (a fixed ``S_CAP``-token sequence,
  no KV cache, no dynamic ``DimVar`` — Phase 0 validates op-composition
  correctness, not context-length scaling), and
- the component -> HF-submodule map.

Component HIR ``@func``s live in ``qwen2_5_1_5b_module.py`` (the ``@module
class`` authoring style, per ``qwen3_1_7b/qwen3_1_7b_module.py``); this
module only holds the shared dims, the HF layer / rope-cache / causal-mask
builders, and the weight-layout helper, so every test file composes them
rather than duplicating the description.

Two structural differences from the ``qwen3_1_7b`` sibling (see
``qwen2_5_1_5b_module.py`` docstring for the HIR-level detail):

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

# ── Qwen2.5-1.5B dimensions ──────────────────────────────────────────────
# Public Qwen2.5-1.5B config.json values (https://huggingface.co/Qwen/Qwen2.5-1.5B):
# hidden_size=1536, num_attention_heads=12, num_key_value_heads=2, head_dim=128,
# intermediate_size=8960, rms_norm_eps=1e-6, rope_theta=1e6,
# max_position_embeddings=32768. Everything below matches those exactly;
# ``VOCAB`` / ``MAX_POS`` already equal the ``Qwen2Config`` dataclass
# defaults (confirmed against the installed ``transformers`` package) and
# are kept explicit only for parity with the ``qwen3_1_7b`` template.
HIDDEN = 1536
HEAD_DIM = 128
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS    # 6 query heads share one kv head
Q_PROJ = NUM_Q_HEADS * HEAD_DIM            # 1536
KV_PROJ = NUM_KV_HEADS * HEAD_DIM          # 256
INTERMEDIATE = 8960
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000.0
VOCAB = 151936
MAX_POS = 32768

# Static test-speed contract: no KV cache, no dynamic dims. This is a
# single-shot prefill oracle (``cur_pos`` is always 0) — a small fixed
# sequence tile is enough since op semantics are length-agnostic; a real
# context length is unnecessary Phase-0 runtime cost.
S_CAP = 4

# HIR dtype for every Tensor annotation in this package: f32 everywhere. There
# is no CUDA on this box, so there is no bf16 branch to also cover.
DT = "f32"

# ── Component -> HF submodule map ───────────────────────────────────────
# Each component's HIR is validated against these submodules of a single
# ``Qwen2DecoderLayer``. ``self_attention`` and ``mlp`` each fuse their
# preceding RMSNorm (see ``qwen2_5_1_5b_module.py`` docstring), so their HF
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
        hidden_size=HIDDEN,
        head_dim=HEAD_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        intermediate_size=INTERMEDIATE,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        num_hidden_layers=1,
        vocab_size=VOCAB,
        max_position_embeddings=MAX_POS,
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
    same ``S_CAP`` tile, i.e. ``cur_pos`` is always 0.
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
