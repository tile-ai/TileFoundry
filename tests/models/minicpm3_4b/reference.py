"""The executable semantics one MiniCPM3-4B decode step is held to.

One step is drawn at a stated context length: the context's hidden states are
drawn, the KV cache is built from them, and the token being decoded is the one
that follows. The cache is passed as plain tensors and the oracle is taken from a
full-sequence forward's last position, so neither side constructs a Hugging Face
cache object. `tests/models/dense_decode.py` owns that drawing, and everything
below states what makes this model's oracle its own.

Everything is seeded, so a disagreement is a disagreement about the compiler
rather than about which random draw each side happened to get.

`residual_scale` is read off the HF layer rather than derived: it is a value the
step is given, not a weight it holds, so it travels after the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch

from tests.models import dense_decode
from tests.models.minicpm3_4b import config

DEVICE = dense_decode.DenseDecode.device
CTX_LEN = dense_decode.DenseDecode.ctx_len


def layer_weights(layer) -> tuple:
    """One layer's weights, in the order its decode step takes them.

    MiniCPM3 attends over a compressed latent: the query and the key/value each
    come from a down-projection, a norm, and an up-projection, which is why this
    list has no single `q_proj`.

    The order is stated here only; both the single-layer arguments and the
    stack's per-layer weights are projected from it.
    """
    attention, mlp = layer.self_attn, layer.mlp
    return (

        layer.input_layernorm.weight,
        config.linear_weight(attention.q_a_proj),
        attention.q_a_layernorm.weight,
        config.linear_weight(attention.q_b_proj),
        config.linear_weight(attention.kv_a_proj_with_mqa),
        attention.kv_a_layernorm.weight,
        config.linear_weight(attention.kv_b_proj),
        config.linear_weight(attention.o_proj),
        layer.post_attention_layernorm.weight,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
    )


def _residual_scale(layer, device: str) -> tuple:
    """The depth-dependent residual scale this step is handed."""
    return (torch.full((1, 1, 1), layer.residual_scale, device=device),)


def _build_decoder():
    from tests.models.minicpm3_4b.decoder import build_decoder  # noqa: PLC0415

    return build_decoder()



DecodeStepInputs = dense_decode.LayerStep
@dataclass(frozen=True)
class DecoderStepInputs(dense_decode.StackStep):
    """A drawn stack step, with its residual scale under its own name."""

    @property
    def residual_scale(self) -> torch.Tensor:
        """The value `trailing` carries, named for what a perturbation test asks."""
        return self.trailing[0]

SPEC = dense_decode.DenseDecode(
    config=config,
    layer_weights=layer_weights,
    attention_weights=7,
    build_decoder=_build_decoder,
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
    "layer_weights",
    "run_decoder_step",
]
