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

DEVICE = dense_decode.DenseDecode.device
CTX_LEN = dense_decode.DenseDecode.ctx_len


def layer_weights(layer) -> tuple:
    """One layer's weights, in the order its decode step takes them.

    Gemma2 norms both sides of the MLP as well as both sides of attention, so it
    carries four norms where the Qwen layers carry two. Its norms are read through
    `config.rms_gamma`, which accounts for Gemma storing gamma as a zero-centred
    offset.

    The order is stated here only; both the single-layer arguments and the
    stack's per-layer weights are projected from it.
    """
    attention, mlp = layer.self_attn, layer.mlp
    return (

        config.rms_gamma(layer.input_layernorm),
        config.linear_weight(attention.q_proj),
        config.linear_weight(attention.k_proj),
        config.linear_weight(attention.v_proj),
        config.linear_weight(attention.o_proj),
        config.rms_gamma(layer.post_attention_layernorm),
        config.rms_gamma(layer.pre_feedforward_layernorm),
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
        config.rms_gamma(layer.post_feedforward_layernorm),
    )


def _build_decoder():
    from tests.models.gemma2_2b.decoder import build_decoder  # noqa: PLC0415

    return build_decoder()



@dataclass(frozen=True)
class DecodeStepInputs(dense_decode.LayerStep):
    """A drawn step, plus the projection `self_attention` takes."""

    @property
    def attention_args(self) -> tuple[torch.Tensor, ...]:
        """`self_attention`'s arguments, derived from the layer's own tuple.

        The attention block here is pure -- it takes already-normalised hidden
        states and no ``gamma_in`` -- so its arguments are the layer's with the
        first replaced by the normed token and the second dropped. Derived rather
        than assembled a second time, so one parameter order is stated once.
        """
        with torch.no_grad():
            normed = self.layer.input_layernorm(self.hidden_new)
        return (normed, *self.args[2:12])
DecoderStepInputs = dense_decode.StackStep

SPEC = dense_decode.DenseDecode(
    config=config,
    layer_weights=layer_weights,
    attention_weights=4,
    build_decoder=_build_decoder,
    layer_step_class=DecodeStepInputs,
    final_norm_of=lambda model: config.rms_gamma(model.norm),
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
    "layer_weights",
    "run_decoder_step",
]
