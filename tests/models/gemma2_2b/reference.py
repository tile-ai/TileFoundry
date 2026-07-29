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

from tests.models import dense_decode
from tests.models.gemma2_2b import config
from tests.models.gemma2_2b.model import Gemma2_2B, Gemma2_2B_Decoder
from tilefoundry.runtime.resource import DictResource

DEVICE = dense_decode.DenseDecode.device
CTX_LEN = dense_decode.DenseDecode.ctx_len


def _layer_constants(layer) -> dict:
    """One layer's weights, keyed the way its Module names them.

    Gemma2 norms both sides of the MLP as well as both sides of attention, so it
    carries four norms where the Qwen layers carry two. Every one of them is read
    through `config.rms_gamma`, which accounts for Gemma storing gamma as a
    zero-centred offset. Stated here rather than shared: which Hugging Face tensor
    a canonical name reads is this model's own fact.
    """
    attention, mlp = layer.self_attn, layer.mlp
    return {
        "gamma_in": config.rms_gamma(layer.input_layernorm),
        "w_q": config.linear_weight(attention.q_proj),
        "w_k": config.linear_weight(attention.k_proj),
        "w_v": config.linear_weight(attention.v_proj),
        "w_o": config.linear_weight(attention.o_proj),
        "gamma_post_attn": config.rms_gamma(layer.post_attention_layernorm),
        "gamma_pre_ff": config.rms_gamma(layer.pre_feedforward_layernorm),
        "w_gate": config.linear_weight(mlp.gate_proj),
        "w_up": config.linear_weight(mlp.up_proj),
        "w_down": config.linear_weight(mlp.down_proj),
        "gamma_post_ff": config.rms_gamma(layer.post_feedforward_layernorm),
    }


def load_layer(layer):
    """The layer Module with *layer*'s weights bound."""
    return Gemma2_2B.cloned().load(DictResource(_layer_constants(layer)))


def load_decoder(model):
    """The decoder root with *model*'s weights bound, one entry per layer.

    ``gamma_final`` goes through `config.rms_gamma` like every other norm here: the
    norm that closes the stack is the same zero-centred offset the four inside a
    layer are, and it is the one where the raw ``.weight`` costs the whole stack's
    output rather than one block's.

    ``w_head`` is supplied in the layout `lm_head` declares: `DictResource` keys are
    already canonical and its converters run in ``prepare``, not here. Reading the
    head off the causal LM rather than deciding from a config field is what makes
    this the same statement for a tied and an untied checkpoint.
    """
    constants = {
        "w_embed": model.model.embed_tokens.weight,
        "gamma_final": config.rms_gamma(model.model.norm),
        "w_head": model.lm_head.weight.t(),
    }
    for index, layer in enumerate(model.model.layers):
        constants.update(
            {f"layer{index}.{name}": w for name, w in _layer_constants(layer).items()}
        )
    return Gemma2_2B_Decoder.cloned().load(DictResource(constants))


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
    config=config,
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
