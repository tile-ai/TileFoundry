"""The executable semantics one MiniCPM3-4B decode step is held to.

One step is drawn at a stated context length: the context's hidden states are
drawn, the KV cache is built from them, and the token being decoded is the one
that follows. The cache is passed as plain tensors and the oracle is taken from a
full-sequence forward's last position, so neither side constructs a Hugging Face
cache object. `tests/models/dense_decode.py` owns that drawing, and everything
below states what makes this model's oracle its own.

The Hugging Face side is built from `model.published()` -- the checkpoint's own
`config.json`, through `MiniCPM3Config` -- so the oracle and the kernels are the
same model by construction rather than by two hand-copied dimension lists
agreeing.

Everything is seeded, so a disagreement is a disagreement about the compiler
rather than about which random draw each side happened to get.

`residual_scale` is read off the HF layer rather than derived: it is a value the
step is given, not a weight it holds, so it travels after the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from tests.models import decode_oracle as oracle
from tests.models import dense_decode
from tests.models.minicpm3_4b.hf_alias import hf_layout_only
from tests.models.minicpm3_4b.model import (
    MiniCPM3_4B,
    MiniCPM3_4B_DecoderLayer,
    config,
    published,
)
from tilefoundry.runtime.resource import DictResource

DEVICE = dense_decode.DenseDecode.device
CTX_LEN = dense_decode.DenseDecode.ctx_len

CONFIG = published()

#: The precision the checkpoint publishes; both sides of every comparison run at it.
DTYPE = CONFIG.dtype



def build_layer(seed=0, device="cpu", dtype=DTYPE):
    """One `MiniCPM3DecoderLayer` with random weights at a fixed seed.

    The published config is passed whole, at its published 62 layers. That is
    visible in the values and not only in the shape: `residual_scale` divides by
    `sqrt(num_hidden_layers)`, so a layer built from a shrunk config would scale
    its residual by 1.4 where the real model scales by 1.4/sqrt(62). The kernel is
    handed the same layer's own `residual_scale`, so both sides move together and
    the number they move to is the published one.
    """
    from transformers.models.minicpm3.modeling_minicpm3 import (  # noqa: PLC0415
        MiniCPM3DecoderLayer,
    )

    return oracle.randomised(
        lambda: MiniCPM3DecoderLayer(CONFIG, layer_idx=0), seed, device, dtype
    )


def build_decoder(seed=0, device="cpu", dtype=DTYPE):
    """The complete published-depth decoder stack, random at a fixed seed.

    A `MiniCPM3ForCausalLM` rather than the base model: the decoder's own boundary
    is still hidden states in and hidden states out, but the root's weights
    include the head, and the head exists only on the causal LM. Its layers and
    final norm are reached through `.model`.
    """
    from transformers.models.minicpm3.modeling_minicpm3 import (  # noqa: PLC0415
        MiniCPM3ForCausalLM,
    )

    return oracle.randomised(lambda: MiniCPM3ForCausalLM(CONFIG), seed, device, dtype)


def _rope_at(rows: int, device="cpu"):
    """Full cos / sin caches `[rows, qk_rope_head_dim]` from the HF rotary
    embedding (`cfg.head_dim`, which `MiniCPM3Config` pins to
    `qk_rope_head_dim == 32`, is the dim it builds caches at).

    Row `p` is the rotary embedding for absolute position `p`, so gathering by
    `pos_ids` reproduces the cos / sin the HF attention applies.
    """
    from transformers.models.minicpm3.modeling_minicpm3 import (  # noqa: PLC0415
        MiniCPM3RotaryEmbedding,
    )

    return oracle.rope_caches(MiniCPM3RotaryEmbedding, CONFIG, rows, device, DTYPE)


def rope_caches(device="cpu"):
    """The caches at the context envelope the kernels are authored for."""
    return _rope_at(config.max_position_embeddings, device)


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
    heads, nope_dim = CONFIG.num_attention_heads, CONFIG.qk_nope_head_dim
    compressed = attention.kv_a_proj_with_mqa(normed)
    latent = compressed[..., : CONFIG.kv_lora_rank]
    rotary_slice = compressed[..., CONFIG.kv_lora_rank :]

    pair = (1, ctx, heads, nope_dim + CONFIG.v_head_dim)
    up = attention.kv_b_proj(attention.kv_a_layernorm(latent)).view(pair).transpose(1, 2)
    nope = up[..., :nope_dim]
    value = up[..., nope_dim:]

    shared = rotary_slice.view(1, 1, ctx, CONFIG.qk_rope_head_dim)
    shared = shared.expand(1, heads, ctx, CONFIG.qk_rope_head_dim)
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

    split = -CONFIG.qk_rope_head_dim
    q_rope, k_rope = apply_rotary_pos_emb(
        query[..., split:], key[..., split:], cos, sin
    )
    return (
        torch.cat([query[..., :split], q_rope], dim=-1),
        torch.cat([key[..., :split], k_rope], dim=-1),
    )


def context_kv(layer, hidden_ctx, device="cpu"):
    """The KV cache *layer* would hold for *hidden_ctx*, as explicit tensors."""
    cos, sin = _rope_at(hidden_ctx.shape[1], device)
    return oracle.context_kv(
        layer, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary,
    )


def decode_reference(layer, hidden_ctx, hidden_new):
    """Hugging Face's output for *hidden_new* decoded after *hidden_ctx*."""
    cos, sin = _rope_at(
        hidden_ctx.shape[1] + hidden_new.shape[1], hidden_ctx.device.type
    )
    return oracle.decode_reference([layer], hidden_ctx, hidden_new, cos, sin)


def decoder_context_kv(model, hidden_ctx, device="cpu"):
    """Per-layer ``(k_cache, v_cache)`` for *hidden_ctx*, in layer order."""
    cos, sin = _rope_at(hidden_ctx.shape[1], device)
    return oracle.stack_context_kv(
        model.model.layers, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary,
    )


def decoder_decode_reference(model, hidden_ctx, hidden_new):
    """The decoder stack's output for *hidden_new* decoded after *hidden_ctx*."""
    cos, sin = _rope_at(
        hidden_ctx.shape[1] + hidden_new.shape[1], hidden_ctx.device.type
    )
    return oracle.decode_reference(
        model.model.layers, hidden_ctx, hidden_new, cos, sin,
        final_norm=model.model.norm,
    )



def _layer_constants(layer) -> dict:
    """One layer's weights, keyed the way its Module names them.

    MiniCPM3 attends over a compressed latent: the query and the key/value each
    come from a down-projection, a norm, and an up-projection, which is why this
    mapping has no single `w_q`. Stated here rather than shared: which Hugging Face
    tensor a canonical name reads is this model's own fact.
    """
    attention, mlp = layer.self_attn, layer.mlp
    return {
        "gamma_in": layer.input_layernorm.weight,
        "w_q_a": attention.q_a_proj.weight,
        "gamma_q_a": attention.q_a_layernorm.weight,
        "w_q_b": attention.q_b_proj.weight,
        "w_kv_a": attention.kv_a_proj_with_mqa.weight,
        "gamma_kv_a": attention.kv_a_layernorm.weight,
        "w_kv_b": attention.kv_b_proj.weight,
        "w_o": attention.o_proj.weight,
        "gamma_post": layer.post_attention_layernorm.weight,
        "w_gate": mlp.gate_proj.weight,
        "w_up": mlp.up_proj.weight,
        "w_down": mlp.down_proj.weight,
    }


def load_layer(layer):
    """The layer Module with *layer*'s weights bound."""
    return MiniCPM3_4B_DecoderLayer.cloned().load(
        DictResource(_layer_constants(layer), alias=hf_layout_only(CONFIG))
    )


def load_decoder(model):
    """The decoder root with *model*'s weights bound, one entry per layer.

    The raw HF layout is supplied throughout; the shipped alias table changes it
    into each declaration's layout. Reading the head off the causal LM rather
    than deciding from a config field makes this the same statement for tied and
    untied checkpoints.
    """
    constants = {
        "w_embed": model.model.embed_tokens.weight,
        "gamma_final": model.model.norm.weight,
        "w_head": model.lm_head.weight,
    }
    for index, layer in enumerate(model.model.layers):
        constants.update(
            {f"layer{index}.{name}": w for name, w in _layer_constants(layer).items()}
        )
    return MiniCPM3_4B.cloned().load(DictResource(constants, alias=hf_layout_only(CONFIG)))


def _residual_scale(layer, device: str) -> tuple:
    """The depth-dependent residual scale this step is handed."""
    return (torch.full((1, 1, 1), layer.residual_scale, device=device),)



DecodeStepInputs = dense_decode.LayerStep


@dataclass(frozen=True)
class DecoderStepInputs(dense_decode.StackStep):
    """A drawn stack step, with its residual scale under its own name."""

    @property
    def residual_scale(self) -> torch.Tensor:
        """The value `trailing` carries, named for what a perturbation test asks."""
        return self.trailing[0]


SPEC = dense_decode.DenseDecode(
    hidden_size=CONFIG.hidden_size,
    dtype=DTYPE,
    rope_caches=rope_caches,
    build_layer=build_layer,
    build_decoder=build_decoder,
    context_kv=context_kv,
    decode_reference=decode_reference,
    decoder_context_kv=decoder_context_kv,
    decoder_decode_reference=decoder_decode_reference,
    load_layer=load_layer,
    load_decoder=load_decoder,
    trailing=_residual_scale,
    stack_step_class=DecoderStepInputs,
)

decode_step_inputs = partial(dense_decode.layer_step, SPEC)
decode_step_oracle = partial(dense_decode.layer_oracle, SPEC)
appended_cache_oracle = partial(dense_decode.appended_cache, SPEC)
decoder_step_inputs = partial(dense_decode.stack_step, SPEC)
run_decoder_step = partial(dense_decode.run_stack, SPEC)
decoder_step_oracle = partial(dense_decode.stack_oracle, SPEC)

__all__ = [
    "CTX_LEN",
    "DEVICE",
    "SPEC",
    "DecodeStepInputs",
    "DecoderStepInputs",
    "appended_cache_oracle",
    "decode_step_inputs",
    "decode_step_oracle",
    "decoder_step_inputs",
    "decoder_step_oracle",
    "load_decoder",
    "load_layer",
    "run_decoder_step",
]
