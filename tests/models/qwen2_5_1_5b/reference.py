"""The executable semantics one Qwen2.5-1.5B decode step is held to.

One step is drawn at a stated context length: the context's hidden states are
drawn, the KV cache is built from them, and the token being decoded is the one
that follows. The cache is passed as plain tensors and the oracle is taken from a
full-sequence forward's last position, so neither side constructs a Hugging Face
cache object. `tests/models/dense_decode.py` owns that drawing, and everything
below states what makes this model's oracle its own.

The Hugging Face side is built from `model.published()` -- the checkpoint's own
`config.json`, through `Qwen2Config` -- so the oracle and the kernels are the same
model by construction rather than by two hand-copied dimension lists agreeing.

Everything is seeded, so a disagreement is a disagreement about the compiler
rather than about which random draw each side happened to get.
"""

from __future__ import annotations

from functools import partial

from tests.models import decode_oracle as oracle
from tests.models import dense_decode
from tests.models.qwen2_5_1_5b.model import (
    HEAD_DIM,
    MAX_CTX,
    Qwen2_5_1_5B,
    Qwen2_5_1_5B_Decoder,
    published,
)
from tilefoundry.runtime.resource import DictResource

DEVICE = dense_decode.DenseDecode.device
CTX_LEN = dense_decode.DenseDecode.ctx_len

CONFIG = published()

#: The precision the checkpoint publishes; both sides of every comparison run at it.
DTYPE = CONFIG.dtype



def build_layer(seed=0, device="cpu", dtype=DTYPE):
    """One `Qwen2DecoderLayer` with random weights at a fixed seed.

    The published config is passed whole. A layer reads only per-layer fields, so
    nothing here has to shrink `num_hidden_layers` first to avoid building 28.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer  # noqa: PLC0415

    return oracle.randomised(
        lambda: Qwen2DecoderLayer(CONFIG, layer_idx=0), seed, device, dtype
    )


def build_decoder(seed=0, device="cpu", dtype=DTYPE):
    """The complete published-depth decoder stack, random at a fixed seed.

    A `Qwen2ForCausalLM` rather than the base model: the decoder's own boundary is still
    hidden states in and hidden states out, but the root's weights include the
    head, and the head exists only on the causal LM. Its layers and final norm are
    reached through `.model`. Stacking one layer's verified behaviour is not the
    same as the stack behaving, which is why this exists separately from
    `build_layer`: layer order, the final norm, and the residual thread between
    layers are only observable here.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM  # noqa: PLC0415

    return oracle.randomised(lambda: Qwen2ForCausalLM(CONFIG), seed, device, dtype)


def _rope_at(rows: int, device="cpu"):
    """Full cos / sin caches `[rows, head_dim]` from the HF rotary embedding.

    Row `p` is the rotary embedding for absolute position `p`, so gathering by
    `pos_ids` reproduces the cos / sin the HF attention applies.
    """
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding  # noqa: PLC0415

    return oracle.rope_caches(Qwen2RotaryEmbedding, CONFIG, rows, device, DTYPE)


def rope_caches(device="cpu"):
    """The caches at the context envelope the kernels are authored for."""
    return _rope_at(MAX_CTX, device)


def _key_value_of(layer, normed):
    """*layer*'s pre-rotary key and its value, head-major.

    The one step of the oracle that is Qwen2's own, and it differs from Qwen3 in
    both directions: there is no per-head norm on the key, and the projections
    carry a bias -- which needs no handling here only because it is inside the
    ``nn.Linear`` this calls.
    """
    attention = layer.self_attn
    heads = (1, normed.shape[1], CONFIG.num_key_value_heads, HEAD_DIM)
    key = attention.k_proj(normed).view(heads).transpose(1, 2)
    value = attention.v_proj(normed).view(heads).transpose(1, 2)
    return key, value



def _apply_rotary():
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb  # noqa: PLC0415

    return apply_rotary_pos_emb


def context_kv(layer, hidden_ctx, device="cpu"):
    """The KV cache *layer* would hold for *hidden_ctx*, as explicit tensors."""
    cos, sin = _rope_at(hidden_ctx.shape[1], device)
    return oracle.context_kv(
        layer, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
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
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
    )


def decoder_decode_reference(model, hidden_ctx, hidden_new):
    """The decoder stack's output for *hidden_new* decoded after *hidden_ctx*."""
    cos, sin = _rope_at(
        hidden_ctx.shape[1] + hidden_new.shape[1], hidden_ctx.device.type
    )
    return oracle.decode_reference(
        model.model.layers, hidden_ctx, hidden_new, cos, sin, final_norm=model.model.norm
    )



def _layer_constants(layer) -> dict:
    """One layer's weights, keyed the way its Module names them.

    Qwen2.5 carries a bias on each attention projection, which is the difference
    between this mapping and Qwen3's -- that one has a per-head query and key norm
    instead. Stated here rather than shared: which Hugging Face tensor a canonical
    name reads is this model's own fact.
    """
    attention, mlp = layer.self_attn, layer.mlp
    return {
        "gamma_in": layer.input_layernorm.weight,
        "w_q": oracle.linear_weight(attention.q_proj),
        "bias_q": attention.q_proj.bias,
        "w_k": oracle.linear_weight(attention.k_proj),
        "bias_k": attention.k_proj.bias,
        "w_v": oracle.linear_weight(attention.v_proj),
        "bias_v": attention.v_proj.bias,
        "w_o": oracle.linear_weight(attention.o_proj),
        "gamma_post": layer.post_attention_layernorm.weight,
        "w_gate": oracle.linear_weight(mlp.gate_proj),
        "w_up": oracle.linear_weight(mlp.up_proj),
        "w_down": oracle.linear_weight(mlp.down_proj),
    }


def load_layer(layer):
    """The layer Module with *layer*'s weights bound."""
    return Qwen2_5_1_5B.cloned().load(DictResource(_layer_constants(layer)))


def load_decoder(model):
    """The decoder root with *model*'s weights bound, one entry per layer.

    ``w_head`` is supplied in the layout `lm_head` declares: `DictResource` keys are
    already canonical and its converters run in ``prepare``, not here. Reading the
    head off the causal LM rather than deciding from a config field is what makes
    this the same statement for a tied and an untied checkpoint.
    """
    constants = {
        "w_embed": model.model.embed_tokens.weight,
        "gamma_final": model.model.norm.weight,
        "w_head": model.lm_head.weight.t(),
    }
    for index, layer in enumerate(model.model.layers):
        constants.update(
            {f"layer{index}.{name}": w for name, w in _layer_constants(layer).items()}
        )
    return Qwen2_5_1_5B_Decoder.cloned().load(DictResource(constants))


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
)

DecodeStepInputs = dense_decode.LayerStep
DecoderStepInputs = dense_decode.StackStep

decode_step_inputs = partial(dense_decode.layer_step, SPEC)
decode_step_oracle = partial(dense_decode.layer_oracle, SPEC)
appended_cache_oracle = partial(dense_decode.appended_cache, SPEC)
decoder_step_inputs = partial(dense_decode.stack_step, SPEC)
run_decoder_step = partial(dense_decode.run_stack, SPEC)
decoder_step_oracle = partial(dense_decode.stack_oracle, SPEC)

__all__ = [
    "CONFIG",
    "CTX_LEN",
    "DEVICE",
    "SPEC",
    "DecodeStepInputs",
    "DecoderStepInputs",
    "appended_cache_oracle",
    "build_decoder",
    "build_layer",
    "decode_step_inputs",
    "decode_step_oracle",
    "decoder_step_inputs",
    "decoder_step_oracle",
    "load_decoder",
    "load_layer",
    "run_decoder_step",
]
