"""The executable semantics one Qwen3-1.7B decode step is held to.

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

from functools import partial

from tests.models import dense_decode
from tests.models.qwen3_1_7b import config
from tests.models.qwen3_1_7b.model import Qwen3_1_7B, Qwen3_1_7B_Decoder
from tilefoundry.runtime.resource import DictResource

DEVICE = dense_decode.DenseDecode.device
CTX_LEN = dense_decode.DenseDecode.ctx_len


def _layer_constants(layer) -> dict:
    """One layer's weights, keyed the way its Module names them.

    Qwen3 carries a per-head query and key norm, which is the difference between
    this mapping and Qwen2.5's -- that one has a bias on each projection instead.
    Stated here rather than shared: which Hugging Face tensor a canonical name
    reads is this model's own fact.
    """
    attention, mlp = layer.self_attn, layer.mlp
    return {
        "gamma_in": layer.input_layernorm.weight,
        "w_q": config.linear_weight(attention.q_proj),
        "w_k": config.linear_weight(attention.k_proj),
        "w_v": config.linear_weight(attention.v_proj),
        "gamma_q": attention.q_norm.weight,
        "gamma_k": attention.k_norm.weight,
        "w_o": config.linear_weight(attention.o_proj),
        "gamma_post": layer.post_attention_layernorm.weight,
        "w_gate": config.linear_weight(mlp.gate_proj),
        "w_up": config.linear_weight(mlp.up_proj),
        "w_down": config.linear_weight(mlp.down_proj),
    }


def load_layer(layer):
    """The layer Module with *layer*'s weights bound."""
    return Qwen3_1_7B.cloned().load(DictResource(_layer_constants(layer)))


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
    return Qwen3_1_7B_Decoder.cloned().load(DictResource(constants))


SPEC = dense_decode.DenseDecode(
    config=config,
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
