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

# ── Qwen3-1.7B dimensions ────────────────────────────────────────────────
# Public Qwen3-1.7B config.json values (https://huggingface.co/Qwen/Qwen3-1.7B):
# hidden_size=2048, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
# intermediate_size=6144, rms_norm_eps=1e-6, rope_theta=1e6,
# max_position_embeddings=32768. Fields the model does not pin (attention_bias)
# fall back to the ``Qwen3Config`` default, which is also ``False``.


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
    max_pos=32768,
    max_ctx=32768,
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


def context_kv(layer, hidden_ctx, device="cpu"):
    """The KV cache for *hidden_ctx*, as the explicit tensors a decode step takes.

    Built by running the layer's own norm, projections and rotary embedding over
    the context — no ``Cache`` object is constructed. What comes out is bitwise
    identical to what Hugging Face's own cache would hold after a prefill of the
    same hidden states, which is what makes it a reference rather than a
    re-derivation: keys are post-rotary (a stored key belongs to the position it
    was written at) and values are post-projection.

    Returned in the kernels' ``[1, ctx_len, n_kv_heads, head_dim]`` layout, not
    Hugging Face's head-major one.
    """
    import torch  # noqa: PLC0415
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb  # noqa: PLC0415

    cfg = build_hf_config()
    attn = layer.self_attn
    ctx = hidden_ctx.shape[1]
    cos, sin = rope_caches(cfg, ctx, device=device)
    with torch.no_grad():
        normed = layer.input_layernorm(hidden_ctx)
        heads = (1, ctx, cfg.num_key_value_heads, cfg.head_dim)
        k = attn.k_norm(attn.k_proj(normed).view(heads)).transpose(1, 2)
        v = attn.v_proj(normed).view(heads).transpose(1, 2)
        # apply_rotary_pos_emb rotates a query/key pair; only the key is wanted.
        _, k = apply_rotary_pos_emb(k, k, cos.unsqueeze(0), sin.unsqueeze(0))
    return k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()


def decode_reference(layer, hidden_ctx, hidden_new, device="cpu"):
    """Hugging Face's output for *hidden_new* decoded after *hidden_ctx*.

    Runs the layer once over the whole sequence under a causal mask and keeps the
    last position. Causality makes that position's output depend on exactly the
    context before it, so this equals a cached one-token step to floating-point
    rounding while touching none of the caching machinery -- the alternative
    would be to hand Hugging Face a ``past_key_values``, and then the reference
    would share the mechanism the kernels are supposed to be checked against.
    """
    import torch  # noqa: PLC0415

    cfg = build_hf_config()
    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    cos, sin = rope_caches(cfg, total, device=device)
    positions = torch.arange(total, device=device)
    mask = torch.where(
        positions.unsqueeze(0) <= positions.unsqueeze(1), 0.0, float("-inf")
    ).view(1, 1, total, total)
    with torch.no_grad():
        out = layer(
            torch.cat([hidden_ctx, hidden_new], dim=1),
            position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=mask,
        )
    return out[:, hidden_ctx.shape[1] :, :]


def linear_weight(linear):
    """HF ``nn.Linear.weight`` ``[out, in]`` -> kernel matmul layout
    ``[1, in, out]`` (the kernel convention is ``x[1,S,in] @ w[1,in,out]``,
    the transpose/pack "weight preprocessing" the task calls for happening in
    test code, not in the kernel)."""
    return linear.weight.t().unsqueeze(0).contiguous()


def build_hf_decoder(seed=0, device="cpu", dtype=None, shape: Qwen3Shape = REAL):
    """The complete ``shape.n_layers``-layer decoder stack, random at a fixed seed.

    A ``Qwen3Model`` is built for its layers and its final norm; its token
    embedding is not part of what this returns, because the decoder's boundary is
    hidden states in and hidden states out. Stacking one layer's verified
    behaviour is not the same as the stack behaving, which is why this exists
    separately from ``build_hf_layer``: layer order, the final norm, and the
    residual thread between layers are only observable here.
    """
    import torch  # noqa: PLC0415
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model  # noqa: PLC0415

    cfg = build_hf_config(shape, layers=shape.n_layers)
    torch.manual_seed(seed)
    model = Qwen3Model(cfg).to(device).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.05)
    if dtype is not None:
        model = model.to(dtype)
    return model


def layer_inputs_over_context(model, hidden_ctx):
    """Each layer's own input hidden states for *hidden_ctx*, in layer order.

    Layer ``i``'s cache is built from what layer ``i`` reads, which is what
    layers ``0..i-1`` produced -- so the context has to be run through the stack
    to know it. Captured with forward-pre-hooks rather than by asking the model
    to keep a cache, so the decode contract holds here too: nothing on either
    side of the comparison constructs a ``past_key_values``.
    """
    import torch  # noqa: PLC0415

    captured: list = [None] * len(model.layers)

    def record(index):
        def hook(_module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            captured[index] = hidden.detach()
            return None
        return hook

    handles = [
        layer.register_forward_pre_hook(record(index), with_kwargs=True)
        for index, layer in enumerate(model.layers)
    ]
    try:
        total = hidden_ctx.shape[1]
        cos, sin = rope_caches(build_hf_config(), total, device=hidden_ctx.device.type)
        positions = torch.arange(total, device=hidden_ctx.device)
        mask = torch.where(
            positions.unsqueeze(0) <= positions.unsqueeze(1), 0.0, float("-inf")
        ).view(1, 1, total, total).to(hidden_ctx.dtype)
        with torch.no_grad():
            _run_layers(model, hidden_ctx, cos, sin, mask)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def _run_layers(model, hidden, cos, sin, mask):
    """*hidden* through every layer and the final norm, cache-free."""
    for layer in model.layers:
        hidden = layer(
            hidden,
            position_embeddings=(cos.unsqueeze(0), sin.unsqueeze(0)),
            attention_mask=mask,
        )
    return model.norm(hidden)


def decoder_context_kv(model, hidden_ctx, device="cpu"):
    """Per-layer ``(k_cache, v_cache)`` for *hidden_ctx*, in layer order."""
    return [
        context_kv(layer, layer_input, device=device)
        for layer, layer_input in zip(
            model.layers, layer_inputs_over_context(model, hidden_ctx)
        )
    ]


def decoder_decode_reference(model, hidden_ctx, hidden_new):
    """The decoder stack's output for *hidden_new* decoded after *hidden_ctx*.

    The whole sequence through every layer and the final norm, last position
    kept -- the same construction the single-layer reference uses, for the same
    reason, and equally free of Hugging Face's caching machinery.
    """
    import torch  # noqa: PLC0415

    device = hidden_ctx.device.type
    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    positions = torch.arange(total, device=hidden_ctx.device)
    mask = torch.where(
        positions.unsqueeze(0) <= positions.unsqueeze(1), 0.0, float("-inf")
    ).view(1, 1, total, total).to(hidden_ctx.dtype)
    with torch.no_grad():
        out = _run_layers(
            model, torch.cat([hidden_ctx, hidden_new], dim=1), cos, sin, mask
        )
    return out[:, hidden_ctx.shape[1] :, :]
