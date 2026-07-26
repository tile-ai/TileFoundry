"""Shared fixtures for the MiniCPM3-4B single decoder-layer HIR description.

Phase 0 "打样": the third of four planned dense/near-dense models (Qwen3-1.7B,
Qwen2.5-1.5B, MiniCPM3-4B, Gemma-2) authored against the same three-file
template as ``tests/models/qwen3_1_7b/`` (``common.py`` +
``<model>_module.py`` + ``test_<model>_module.py``), run on macOS with no
CUDA: **cpu + f32 only**.

Unlike the Qwen3 siblings (plain GQA), MiniCPM3 uses **Multi-head Latent
Attention (MLA)**: queries are a low-rank down/up projection
(``q_a_proj`` -> ``q_a_layernorm`` -> ``q_b_proj``), keys/values come from a
*shared* low-rank latent (``kv_a_proj_with_mqa`` -> ``kv_a_layernorm`` ->
``kv_b_proj``) that up-projects to one distinct (nope, value) pair *per query
head*, and RoPE applies only to a narrow rotary slice
(``qk_rope_head_dim=32``) that is carved out of the query/key head dim before
the up-projections and spliced back in afterwards — the "nope" (no positional
encoding) slice never rotates. See ``minicpm3_4b_module.py`` for the full
7-step op-level breakdown, matching
``transformers.models.minicpm3.modeling_minicpm3.MiniCPM3Attention.forward``.

MiniCPM3 also scales its residual branches by ``scale_depth /
sqrt(num_hidden_layers)`` (a muP-style depth scaling MiniCPM calls
``scale_depth``; see ``MiniCPM3DecoderLayer.forward``) instead of adding the
branch directly (contrast Qwen3/Llama). ``scale_emb`` (input-embedding scale)
and ``dim_model_base`` (``logits_scaling``, applied before the LM head) both
live outside a single decoder layer's forward (embedding lookup / lm_head are
not part of ``MiniCPM3DecoderLayer``), so neither is needed here.

Dimensions below are the real ``MiniCPM3Config()`` defaults (which are the
``openbmb/MiniCPM3-4B`` checkpoint dimensions) — confirmed via a REPL
(``MiniCPM3Config()``) rather than assumed:

- ``hidden_size=2560, intermediate_size=6400, num_attention_heads=40,
  num_key_value_heads=40`` (so ``num_key_value_groups = 40 // 40 == 1``: MLA's
  ``kv_b_proj`` already produces one distinct nope/value pair per query head,
  so — unlike Qwen3's GQA — there is no cross-head repeat for the nope/value
  parts; the *only* cross-head broadcast in this whole model is the shared
  rotary slice of K, step 5 below),
- ``q_lora_rank=768, kv_lora_rank=256, qk_nope_head_dim=64,
  qk_rope_head_dim=32`` (so ``qk_head_dim = 64 + 32 == 96``),
  ``v_head_dim=64`` (defaults to ``hidden_size // num_attention_heads`` when
  unset, i.e. ``2560 // 40``; note ``num_attention_heads * v_head_dim ==
  hidden_size`` here, ``40*64==2560``, purely a property of this config, not
  a general MLA identity),
- ``rms_norm_eps=1e-5`` for ``input_layernorm`` / ``post_attention_layernorm``
  — **but** ``q_a_layernorm`` / ``kv_a_layernorm`` are each a bare
  ``MiniCPM3RMSNorm(dim)`` with no explicit ``eps=`` kwarg, so those two use
  ``MiniCPM3RMSNorm.__init__``'s own default (``1e-6``), not
  ``config.rms_norm_eps`` — confirmed by reading both the eps each submodule
  reports at construction time and the source (``modeling_minicpm3.py``);
  this package therefore carries *two* rms-eps constants (``RMS_EPS`` vs
  ``RMS_EPS_LORA``), not one. This is not just cosmetic: measured directly
  (plain-torch rms_norm on realistically-scaled activations at this model's
  dims), ``eps=1e-6`` vs ``eps=1e-5`` differ by ~2.6e-4 absolute — bigger than
  this package's ``atol=2e-4`` — so using the wrong eps here is not
  guaranteed to be masked by the test tolerance,
- ``rope_theta=10000.0`` lives in ``config.rope_parameters['rope_theta']``
  (transformers 5.14.1 rope-params-as-dict convention); the rotary head dim
  actually used is ``config.head_dim``, which ``MiniCPM3Config.__post_init__``
  unconditionally sets to ``qk_rope_head_dim`` (**32**, not
  ``hidden_size / num_attention_heads``) — so the RoPE cos/sin cache is
  ``[max_pos, 32]``, matching only the rotary slice, never the full
  ``qk_head_dim=96``,
- ``scale_depth=1.4``; ``MiniCPM3DecoderLayer.__init__`` precomputes
  ``residual_scale = scale_depth / sqrt(num_hidden_layers)`` once. This
  package builds its HF config/layer with ``num_hidden_layers=1`` (a single
  decoder layer, mirroring the Qwen3 siblings' ``num_hidden_layers=1`` —
  there is no checkpoint to match numerically, only random weights at a fixed
  seed), so ``residual_scale == 1.4 / sqrt(1) == 1.4`` for every fixture in
  this package; the tests read ``layer.residual_scale`` directly off the HF
  layer rather than re-deriving the formula, so this package has no
  hardcoded dependence on ``num_hidden_layers == 1`` beyond that config
  choice itself.

Static test-speed contract (matching every Phase-0 sibling): a fixed
``S_CAP``-token sequence, no KV cache, no dynamic ``DimVar``, cpu + f32.
"""
from __future__ import annotations

# ── MiniCPM3-4B dimensions (== MiniCPM3Config() defaults; see module
# docstring above for the REPL-confirmed values and the two rms-eps /
# rope-head-dim / scale_depth gotchas) ───────────────────────────────────
HIDDEN = 2560
INTERMEDIATE = 6400
NUM_Q_HEADS = 40
NUM_KV_HEADS = 40                          # num_key_value_groups == 1 (see above)
Q_LORA_RANK = 768
KV_LORA_RANK = 256
QK_NOPE_HEAD_DIM = 64
QK_ROPE_HEAD_DIM = 32
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM   # 96
V_HEAD_DIM = 64
RMS_EPS = 1e-5                              # input_layernorm / post_attention_layernorm
RMS_EPS_LORA = 1e-6                         # q_a_layernorm / kv_a_layernorm (MiniCPM3RMSNorm default)
ROPE_THETA = 10_000.0
ATTENTION_BIAS = False
VOCAB = 73448
MAX_POS = 32768
SCALE_DEPTH = 1.4

# Weight fan-out widths (kernel-layout second dim; see `linear_weight` below):
Q_UP_PROJ = NUM_Q_HEADS * QK_HEAD_DIM                       # q_b_proj out: 3840
KV_A_PROJ = KV_LORA_RANK + QK_ROPE_HEAD_DIM                 # kv_a_proj_with_mqa out: 288
KV_B_PROJ = NUM_Q_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)   # kv_b_proj out: 5120
ATTN_OUT = NUM_Q_HEADS * V_HEAD_DIM                         # o_proj in: 2560 (== HIDDEN here)

# Static test-speed contract: no KV cache, no dynamic dims — a single-shot
# prefill oracle (`cur_pos` is always 0); op semantics are length-agnostic so
# a small fixed tile is enough (mirrors every Phase-0 sibling's `S_CAP`).
S_CAP = 4

# HIR dtype for every Tensor annotation in this package: f32 everywhere (no
# CUDA on this box, so no bf16 branch to also cover).
DT = "f32"

# ── Component -> HF submodule map ───────────────────────────────────────
# `mla_attention` and `mlp` each fuse their preceding RMSNorm (see
# `minicpm3_4b_module.py` docstring), so their HF comparison composes the
# norm + block rather than the block alone — matching the Qwen3-1.7B
# sibling's convention exactly.
COMPONENT_HF_SUBMODULES = {
    "input_rms_norm": ("input_layernorm",),
    "mla_attention": ("input_layernorm", "self_attn"),
    "mlp": ("post_attention_layernorm", "mlp"),
    "decoder_layer": (".",),
}


def build_hf_config():
    """Build a ``MiniCPM3Config`` at the MiniCPM3-4B dimensions (one decoder
    layer). Every field is spelled out explicitly (even where it matches the
    class default) so this config is self-documenting and does not silently
    drift if upstream defaults change."""
    from transformers import MiniCPM3Config  # noqa: PLC0415

    return MiniCPM3Config(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        num_hidden_layers=1,
        rms_norm_eps=RMS_EPS,
        rope_theta=ROPE_THETA,
        attention_bias=ATTENTION_BIAS,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        scale_depth=SCALE_DEPTH,
        vocab_size=VOCAB,
        max_position_embeddings=MAX_POS,
    )


def build_hf_layer(seed=0, device="cpu", dtype=None):
    """Build a ``MiniCPM3DecoderLayer`` with random weights at a fixed seed.

    ``device`` defaults to ``"cpu"`` (no CUDA on this box — every caller in
    this package either omits ``device`` or passes ``"cpu"`` explicitly).
    """
    import torch  # noqa: PLC0415
    from transformers.models.minicpm3.modeling_minicpm3 import MiniCPM3DecoderLayer  # noqa: PLC0415

    cfg = build_hf_config()
    torch.manual_seed(seed)
    layer = MiniCPM3DecoderLayer(cfg, layer_idx=0).to(device).eval()
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0.0, 0.05)
    if dtype is not None:
        layer = layer.to(dtype)
    return layer


def rope_caches(cfg, max_pos, device="cpu", dtype=None):
    """Full cos / sin caches ``[max_pos, qk_rope_head_dim]`` from the HF
    rotary embedding (``config.head_dim`` — which ``MiniCPM3Config`` pins to
    ``qk_rope_head_dim==32``, not the full ``qk_head_dim==96`` — is the dim
    ``MiniCPM3RotaryEmbedding`` actually builds caches at).

    Row ``p`` is the rotary embedding for absolute position ``p``, so
    gathering by ``pos_ids`` reproduces the cos / sin the HF attention
    applies.
    """
    import torch  # noqa: PLC0415
    from transformers.models.minicpm3.modeling_minicpm3 import (  # noqa: PLC0415
        MiniCPM3RotaryEmbedding,
    )

    rotary = MiniCPM3RotaryEmbedding(cfg).to(device)
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

    This package has no KV cache: every token attends only within the same
    ``S_CAP`` tile, i.e. ``cur_pos`` is always 0.
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
    the transpose/pack "weight preprocessing" happening in test code, not in
    the kernel)."""
    return linear.weight.t().unsqueeze(0).contiguous()
