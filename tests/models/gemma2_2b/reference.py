"""The executable semantics one Gemma2-2B decode step is held to.

One step is drawn at a stated context length: the context's hidden states are
drawn, the KV cache is built from them, and the token being decoded is the one
that follows. The cache is passed as plain tensors and the oracle is taken from a
full-sequence forward's last position, so neither side constructs a Hugging Face
cache object. `tests/models/dense_decode.py` owns that drawing, and everything
below states what makes this model's oracle its own.

Everything is seeded, so a disagreement is a disagreement about the compiler
rather than about which random draw each side happened to get.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from tests.models import decode_oracle as oracle
from tests.models import dense_decode
from tests.models.gemma2_2b.hf_alias import hf_layout_only
from tests.models.gemma2_2b.model import (
    Gemma2_2B,
    Gemma2_2B_DecoderLayer,
    config,
    published,
)
from tilefoundry.runtime.resource import DictResource

DEVICE = dense_decode.DenseDecode.device
CTX_LEN = dense_decode.DenseDecode.ctx_len

#: Eager attention, because Gemma-2 caps attention logits at 50.0 and only the
#: eager path applies the cap; the default would silently drop it.
CONFIG = published(attn_implementation="eager")

#: The precision the checkpoint publishes; both sides of every comparison run at it.
DTYPE = CONFIG.dtype



def _within_window(total: int) -> int:
    """*total* positions, checked against Gemma-2's sliding window.

    Half of Gemma-2's layers attend within ``sliding_window`` rather than over the
    whole context, and these kernels describe full attention. The two agree
    exactly while the window does not bind and not at all once it does, so the
    boundary is enforced rather than documented.
    """
    if total > CONFIG.sliding_window:
        raise ValueError(
            f"{total} positions exceeds Gemma-2's sliding window "
            f"({CONFIG.sliding_window}); half the layers would attend a window "
            f"rather than the whole context, which these kernels do not describe"
        )
    return total


def build_layer(seed=0, device="cpu", dtype=DTYPE):
    """One `Gemma2DecoderLayer` with random weights at a fixed seed.

    The published config is passed whole. A layer reads only per-layer fields, so
    nothing here has to shrink `num_hidden_layers` first to avoid building 26.
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer  # noqa: PLC0415

    return oracle.randomised(
        lambda: Gemma2DecoderLayer(CONFIG, layer_idx=0), seed, device, dtype
    )


def build_decoder(seed=0, device="cpu", dtype=DTYPE):
    """The complete published-depth decoder stack, random at a fixed seed.

    A `Gemma2ForCausalLM` rather than the base model: the decoder's own boundary
    is still hidden states in and hidden states out, but the root's weights
    include the head, and the head exists only on the causal LM. Its layers and
    final norm are reached through `.model`.
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM  # noqa: PLC0415

    return oracle.randomised(lambda: Gemma2ForCausalLM(CONFIG), seed, device, dtype)


def _rope_at(rows: int, device="cpu"):
    """Full cos / sin caches `[rows, head_dim]` from the HF rotary embedding.

    Row `p` is the rotary embedding for absolute position `p`, so gathering by
    `pos_ids` reproduces the cos / sin the HF attention applies.
    """
    from transformers.models.gemma2.modeling_gemma2 import Gemma2RotaryEmbedding  # noqa: PLC0415

    return oracle.rope_caches(Gemma2RotaryEmbedding, CONFIG, rows, device, DTYPE)


def rope_caches(device="cpu"):
    """The caches at the context envelope the kernels are authored for."""
    return _rope_at(config.sliding_window, device)


def _key_value_of(layer, normed):
    """*layer*'s pre-rotary key and its value, head-major.

    The one step of the oracle that is Gemma-2's own, and it is the plainest of
    the corpus: no per-head norm on the key (that is Qwen3's), no projection bias
    (``attention_bias`` is False), just the projection reshaped into heads.
    """
    attention = layer.self_attn
    heads = (1, normed.shape[1], CONFIG.num_key_value_heads, CONFIG.head_dim)
    key = attention.k_proj(normed).view(heads).transpose(1, 2)
    value = attention.v_proj(normed).view(heads).transpose(1, 2)
    return key, value


def _apply_rotary():
    from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb  # noqa: PLC0415

    return apply_rotary_pos_emb


def context_kv(layer, hidden_ctx, device="cpu"):
    """The KV cache *layer* would hold for *hidden_ctx*, as explicit tensors."""
    cos, sin = _rope_at(_within_window(hidden_ctx.shape[1]), device)
    return oracle.context_kv(
        layer, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
    )


def decode_reference(layer, hidden_ctx, hidden_new):
    """Hugging Face's output for *hidden_new* decoded after *hidden_ctx*."""
    total = _within_window(hidden_ctx.shape[1] + hidden_new.shape[1])
    cos, sin = _rope_at(total, hidden_ctx.device.type)
    return oracle.decode_reference([layer], hidden_ctx, hidden_new, cos, sin)


def decoder_context_kv(model, hidden_ctx, device="cpu"):
    """Per-layer ``(k_cache, v_cache)`` for *hidden_ctx*, in layer order."""
    cos, sin = _rope_at(_within_window(hidden_ctx.shape[1]), device)
    return oracle.stack_context_kv(
        model.model.layers, hidden_ctx, cos, sin,
        key_value_of=_key_value_of, apply_rotary=_apply_rotary(),
    )


def decoder_decode_reference(model, hidden_ctx, hidden_new):
    """The decoder stack's output for *hidden_new* decoded after *hidden_ctx*."""
    total = _within_window(hidden_ctx.shape[1] + hidden_new.shape[1])
    cos, sin = _rope_at(total, hidden_ctx.device.type)
    return oracle.decode_reference(
        model.model.layers, hidden_ctx, hidden_new, cos, sin,
        final_norm=model.model.norm,
    )



def _layer_constants(layer) -> dict:
    """One layer's weights, keyed the way its Module names them.

    Gemma2 norms both sides of the MLP as well as both sides of attention, so it
    carries four norms where the Qwen layers carry two. Each is the checkpoint's
    tensor as stored; Gemma keeps gamma as a zero-centred offset and the kernels
    add the one back, so nothing is adjusted on the way in. Stated here rather
    than shared: which Hugging Face tensor a canonical name reads is this model's
    own fact.
    """
    attention, mlp = layer.self_attn, layer.mlp
    return {
        "gamma_in": layer.input_layernorm.weight,
        "w_q": attention.q_proj.weight,
        "w_k": attention.k_proj.weight,
        "w_v": attention.v_proj.weight,
        "w_o": attention.o_proj.weight,
        "gamma_post_attn": layer.post_attention_layernorm.weight,
        "gamma_pre_ff": layer.pre_feedforward_layernorm.weight,
        "w_gate": mlp.gate_proj.weight,
        "w_up": mlp.up_proj.weight,
        "w_down": mlp.down_proj.weight,
        "gamma_post_ff": layer.post_feedforward_layernorm.weight,
    }


def load_layer(layer):
    """The layer Module with *layer*'s weights bound."""
    return Gemma2_2B_DecoderLayer.cloned().load(
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
    return Gemma2_2B.cloned().load(DictResource(constants, alias=hf_layout_only(CONFIG)))


@dataclass(frozen=True)
class DecodeStepInputs(dense_decode.LayerStep):
    """A drawn step, plus the activations `self_attention` takes."""

    @property
    def attention_args(self) -> tuple[torch.Tensor, ...]:
        """`self_attention`'s activations, derived from the layer's own.

        The attention block here is pure -- it takes already-normalised hidden
        states -- so its activations are the layer's with the first replaced by the
        normed token. Derived rather than assembled a second time, so one parameter
        order is stated once.
        """
        with torch.no_grad():
            normed = self.layer.input_layernorm(self.hidden_new)
        return (normed, *self.args[1:])


DecoderStepInputs = dense_decode.StackStep

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
    layer_step_class=DecodeStepInputs,
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
