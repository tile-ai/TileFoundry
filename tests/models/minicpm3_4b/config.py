"""Shared fixtures for the MiniCPM3-4B decoder HIR description.

Pins the value oracle and the model contract every test in this package shares:

- the Hugging Face reference (``transformers`` ``MiniCPM3DecoderLayer`` /
  ``MiniCPM3Model`` built from a ``MiniCPM3Config`` at the MiniCPM3-4B
  dimensions, random weights at a fixed seed),
- the model dimensions,
- the decode contract: one token per step (``seq_len`` is the literal 1), the
  active context length ``ctx_len`` as the single dynamic dimension, and the KV
  cache passed as explicit tensors in and out -- Hugging Face's
  ``past_key_values`` is never constructed, on either side of the comparison,
  and
- the component -> HF-submodule map.

Unlike the Qwen siblings (plain GQA), MiniCPM3 is the corpus's only **Multi-head
Latent Attention (MLA)** model: queries are a low-rank down/up projection
(``q_a_proj`` -> ``q_a_layernorm`` -> ``q_b_proj``), keys and values come from a
*shared* low-rank latent (``kv_a_proj_with_mqa`` -> ``kv_a_layernorm`` ->
``kv_b_proj``) that up-projects to one distinct (nope, value) pair *per query
head*, and RoPE applies only to a narrow rotary slice (``qk_rope_head_dim=32``)
carved out of the query/key head dim -- the "nope" (no positional encoding)
slice never rotates. See ``model/decoder_layer.py`` for the op-level breakdown.

MiniCPM3 also scales each residual branch by ``scale_depth /
sqrt(num_hidden_layers)`` (a muP-style depth scaling MiniCPM calls
``scale_depth``) instead of adding the branch directly. ``scale_emb``
(input-embedding scale) and ``dim_model_base`` (``logits_scaling``, applied
before the LM head) both live outside the decoder's boundary -- hidden states in,
hidden states out -- so neither is needed here.

── What the explicit KV cache holds, and why ────────────────────────────────

A production MLA serving stack caches the *latent*: the ``kv_lora_rank``
compressed vector plus the shared ``qk_rope_head_dim`` rotary slice, 288 numbers
per position rather than 40 heads x 96. Hugging Face does not. Read
``MiniCPM3Attention.forward``: the ``past_key_values.update(...)`` call happens
*after* ``key_states = torch.cat((k_pass, k_rot), dim=-1)`` and after ``k_rot``
has been expanded across all query heads, so what its cache holds is the fully
assembled per-head key ``[1, heads, ctx, qk_nope_head_dim + qk_rope_head_dim]``
and the up-projected value ``[1, heads, ctx, v_head_dim]``.

The explicit tensors hold exactly that, which is what makes the oracle
reproducible without a ``Cache`` object: ``_key_value_of`` runs the same
projections in the same order and reaches the same tensors HF would have handed
to ``update``. Measured rather than argued, and end to end rather than against a
transcription of HF's internals: a decode step reading these tensors reproduces
the last position of a full-sequence causal forward to 3e-7, which a cache
holding anything else -- the latent, say -- could not do at any tolerance.

Two consequences of that choice worth stating. The key's head dim (96) and the
value's (64) differ, so the two caches are not the same shape. And
``num_key_value_heads == num_attention_heads`` here, so the cache carries one
entry per query head and nothing repeats it on the way in; the only cross-head
broadcast left in the model is the shared rotary slice of K, inside the step.

── Dimensions ──────────────────────────────────────────────────────────────

``MiniCPM3Config()``'s own defaults, read off a constructed config rather than
assumed. Those defaults are the ``openbmb/MiniCPM3-4B`` values on upstream's own
say-so, not on an inference from the numbers: ``configuration_minicpm3.py``
carries ``@auto_docstring(checkpoint="openbmb/MiniCPM3-4B")`` over the class and
"Defaults match the ``openbmb/MiniCPM3-4B`` checkpoint" over the field block. No
checkpoint is downloaded here; nothing in this package needs one, because the
weights are random at a fixed seed and only the shape has to be the real one.

Note the floor: ``minicpm3`` first ships in ``transformers`` 5.13.0 and is absent
from 5.12.x, which is what this repo's dependency currently names as its minimum.

``hidden_size=2560``, ``intermediate_size=6400``,
``num_attention_heads=40``, ``num_key_value_heads=40``,
``num_hidden_layers=62``, ``q_lora_rank=768``, ``kv_lora_rank=256``,
``qk_nope_head_dim=64``, ``qk_rope_head_dim=32``, ``v_head_dim=64``,
``rms_norm_eps=1e-5``, ``scale_depth=1.4``, ``vocab_size=73448``,
``max_position_embeddings=32768``.

Three of those need a note:

- ``q_a_layernorm`` / ``kv_a_layernorm`` are each a bare ``MiniCPM3RMSNorm(dim)``
  with no ``eps=`` kwarg, so they use ``MiniCPM3RMSNorm.__init__``'s own default
  ``1e-6`` rather than ``config.rms_norm_eps``. This package therefore carries
  *two* rms-eps constants. Not cosmetic: at this model's dims the two eps differ
  by ~2.6e-4 absolute, wider than this package's ``atol``.
- ``rope_theta`` lives in ``config.rope_parameters['rope_theta']``, and the
  rotary dim is ``config.head_dim``, which ``MiniCPM3Config`` pins to
  ``qk_rope_head_dim`` (**32**, not ``hidden_size / num_attention_heads``). The
  cos/sin caches are ``[max_pos, 32]``, matching only the rotary slice.
- ``residual_scale = scale_depth / sqrt(num_hidden_layers)`` is precomputed per
  layer in ``MiniCPM3DecoderLayer.__init__``, so it differs between a one-layer
  component fixture (1.4) and the real 62-layer stack (1.4/sqrt(62)). Every test
  reads ``layer.residual_scale`` off the HF layer rather than re-deriving it, so
  nothing here depends on which layer count was built.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tests.models import decode_oracle as oracle

# Every dimension below comes from the published configuration, pinned so the claim
# is checkable rather than quoted. Checked: the values in the docstring above agree
# with the file at this revision, `max_position_embeddings` included.
#
#: Where the dimensions come from.
SOURCE_URL = "https://huggingface.co/openbmb/MiniCPM3-4B/blob/main/config.json"
#: The commit the values were read at. A full sha rather than a branch, because a
#: branch names whatever it points at today.
SOURCE_REVISION = "d6b14ddaefdb11c624dd75c3c779549bc90b08cb"
#: sha256 of that file as fetched.
SOURCE_SHA256 = "cf1d08cb7c1815c676e685bd6ce94eb8b85a57d53871e6e159ee8c650717d98a"


@dataclass(frozen=True)
class MiniCPM3Shape:
    """One decoder layer's shape, plus the context envelope and dtype every
    kernel in this package is authored at."""

    hidden: int
    intermediate: int
    n_q_heads: int
    n_kv_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_eps: float
    rms_eps_lora: float
    rope_theta: float
    attention_bias: bool
    vocab: int
    max_pos: int
    max_ctx: int
    n_layers: int
    scale_depth: float
    dt: str

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def q_up_proj(self) -> int:
        """``q_b_proj`` out."""
        return self.n_q_heads * self.qk_head_dim

    @property
    def kv_a_proj(self) -> int:
        """``kv_a_proj_with_mqa`` out."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kv_b_proj(self) -> int:
        """``kv_b_proj`` out."""
        return self.n_q_heads * (self.qk_nope_head_dim + self.v_head_dim)

    @property
    def attn_out(self) -> int:
        """``o_proj`` in."""
        return self.n_q_heads * self.v_head_dim


# One token per step, so the only dynamic dimension is the context the step
# reads. ``max_ctx`` matches ``max_pos``: a position beyond the rotary cache has
# no embedding to gather, so a longer context is unrepresentable rather than slow.
SEQ_LEN = 1

REAL = MiniCPM3Shape(
    hidden=2560,
    intermediate=6400,
    n_q_heads=40,
    n_kv_heads=40,
    q_lora_rank=768,
    kv_lora_rank=256,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    rms_eps=1e-05,
    rms_eps_lora=1e-06,
    rope_theta=10000.0,
    attention_bias=False,
    vocab=73448,
    max_pos=32768,
    max_ctx=32768,
    n_layers=62,
    scale_depth=1.4,
    dt='f32',
)

# ── Component -> HF submodule map ───────────────────────────────────────
# Each component's HIR is validated against these submodules of a single
# ``MiniCPM3DecoderLayer``. ``mla_attention`` and ``mlp`` each fuse their
# preceding RMSNorm (see ``model/decoder_layer.py`` docstring), so their HF
# comparison composes the norm + block rather than the block alone.
COMPONENT_HF_SUBMODULES = {
    "input_rms_norm": ("input_layernorm",),
    "mla_attention": ("input_layernorm", "self_attn"),
    "mlp": ("post_attention_layernorm", "mlp"),
    "decoder_layer": (".",),
}


def build_hf_config(*, layers: int = 1):
    """Build a ``MiniCPM3Config`` at the MiniCPM3-4B dimensions.

    ``layers`` defaults to one, which is what a component test needs. The
    complete-decoder reference asks for ``REAL.n_layers`` instead -- and the
    difference is visible in the values, not only in the shape, because
    ``residual_scale`` divides by ``sqrt(num_hidden_layers)``.

    Every field is spelled out even where it matches the class default, so this
    config does not silently drift if an upstream default changes.
    """
    from transformers import MiniCPM3Config  # noqa: PLC0415

    return MiniCPM3Config(
        hidden_size=REAL.hidden,
        intermediate_size=REAL.intermediate,
        num_attention_heads=REAL.n_q_heads,
        num_key_value_heads=REAL.n_kv_heads,
        num_hidden_layers=layers,
        rms_norm_eps=REAL.rms_eps,
        rope_theta=REAL.rope_theta,
        attention_bias=REAL.attention_bias,
        q_lora_rank=REAL.q_lora_rank,
        kv_lora_rank=REAL.kv_lora_rank,
        qk_nope_head_dim=REAL.qk_nope_head_dim,
        qk_rope_head_dim=REAL.qk_rope_head_dim,
        v_head_dim=REAL.v_head_dim,
        scale_depth=REAL.scale_depth,
        vocab_size=REAL.vocab,
        max_position_embeddings=REAL.max_pos,
    )


def build_hf_layer(seed=0, device="cpu", dtype=None):
    """Build a ``MiniCPM3DecoderLayer`` with random weights at a fixed seed."""
    from transformers.models.minicpm3.modeling_minicpm3 import (  # noqa: PLC0415
        MiniCPM3DecoderLayer,
    )

    return oracle.randomised(
        lambda: MiniCPM3DecoderLayer(build_hf_config(), layer_idx=0), seed, device, dtype
    )


def build_hf_decoder(seed=0, device="cpu", dtype=None):
    """The complete ``REAL.n_layers``-layer decoder stack, random at a fixed seed.

    Built for its layers and its final norm; the token embedding is not part of
    what this returns, because the decoder's boundary is hidden states in and
    hidden states out.
    """
    from transformers.models.minicpm3.modeling_minicpm3 import (  # noqa: PLC0415
        MiniCPM3Model,
    )

    return oracle.randomised(
        lambda: MiniCPM3Model(build_hf_config(layers=REAL.n_layers)), seed, device, dtype
    )


def rope_caches(cfg, max_pos, device="cpu", dtype=None):
    """Full cos / sin caches ``[max_pos, qk_rope_head_dim]`` from the HF rotary
    embedding (``cfg.head_dim``, which ``MiniCPM3Config`` pins to
    ``qk_rope_head_dim == 32``, is the dim it builds caches at).

    Row ``p`` is the rotary embedding for absolute position ``p``, so gathering
    by ``pos_ids`` reproduces the cos / sin the HF attention applies.
    """
    from transformers.models.minicpm3.modeling_minicpm3 import (  # noqa: PLC0415
        MiniCPM3RotaryEmbedding,
    )

    return oracle.rope_caches(MiniCPM3RotaryEmbedding, cfg, max_pos, device, dtype)


def _key_value_of(layer, normed):
    """*layer*'s pre-rotary key and its value, head-major.

    The one step of the oracle that is MiniCPM3's own, and MLA makes it the
    longest in the corpus: the key is not a projection of the hidden states but
    the concatenation of a per-head up-projection of the shared latent (the nope
    half, never rotated) with the latent's own rotary slice broadcast across
    heads. ``MiniCPM3Attention.forward`` assembles it in exactly this order and
    hands the result to its cache, so this is the cache's content by
    construction.

    Broadcasting the rotary slice before rotating rather than after -- HF
    rotates the one shared head then expands -- is the same values either way:
    the rotation depends on position, not on head, so it commutes with a
    broadcast along the head axis.
    """
    attention = layer.self_attn
    ctx = normed.shape[1]
    compressed = attention.kv_a_proj_with_mqa(normed)
    latent = compressed[..., : REAL.kv_lora_rank]
    rotary_slice = compressed[..., REAL.kv_lora_rank :]

    pair = (1, ctx, REAL.n_q_heads, REAL.qk_nope_head_dim + REAL.v_head_dim)
    up = attention.kv_b_proj(attention.kv_a_layernorm(latent)).view(pair).transpose(1, 2)
    nope = up[..., : REAL.qk_nope_head_dim]
    value = up[..., REAL.qk_nope_head_dim :]

    shared = rotary_slice.view(1, 1, ctx, REAL.qk_rope_head_dim)
    shared = shared.expand(1, REAL.n_q_heads, ctx, REAL.qk_rope_head_dim)
    # nope first, rope second: the layout the step's query and key also use.
    return torch.cat([nope, shared], dim=-1), value


def _apply_rotary(query, key, cos, sin):
    """Rotate only the trailing ``qk_rope_head_dim`` of *query* and *key*.

    The oracle's ``context_kv`` rotates a stored key by calling this, and for
    every other model in the corpus that means the whole head. For MLA it means
    the last 32 of 96: the nope slice passes through untouched. Same signature as
    Hugging Face's ``apply_rotary_pos_emb`` so the oracle needs no special case.
    """
    from transformers.models.minicpm3.modeling_minicpm3 import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    split = -REAL.qk_rope_head_dim
    q_rope, k_rope = apply_rotary_pos_emb(
        query[..., split:], key[..., split:], cos, sin
    )
    return (
        torch.cat([query[..., :split], q_rope], dim=-1),
        torch.cat([key[..., :split], k_rope], dim=-1),
    )


def context_kv(layer, hidden_ctx, device="cpu"):
    """The KV cache *layer* would hold for *hidden_ctx*, as explicit tensors."""
    cos, sin = rope_caches(build_hf_config(), hidden_ctx.shape[1], device=device)
    return oracle.context_kv(
        layer, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary,
    )


def decode_reference(layer, hidden_ctx, hidden_new, device="cpu"):
    """Hugging Face's output for *hidden_new* decoded after *hidden_ctx*."""
    total = hidden_ctx.shape[1] + hidden_new.shape[1]
    cos, sin = rope_caches(build_hf_config(), total, device=device)
    return oracle.decode_reference([layer], hidden_ctx, hidden_new, cos, sin)


def decoder_context_kv(model, hidden_ctx, device="cpu"):
    """Per-layer ``(k_cache, v_cache)`` for *hidden_ctx*, in layer order."""
    cos, sin = rope_caches(build_hf_config(), hidden_ctx.shape[1], device=device)
    return oracle.stack_context_kv(
        model.layers, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary,
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
    ``[1, in, out]`` (the kernel convention is ``x[1,S,in] @ w[1,in,out]``, the
    transpose/pack "weight preprocessing" happening in test code, not in the
    kernel)."""
    return oracle.linear_weight(linear)
