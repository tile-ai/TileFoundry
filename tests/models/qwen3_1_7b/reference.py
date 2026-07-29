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

DEVICE = dense_decode.DenseDecode.device
CTX_LEN = dense_decode.DenseDecode.ctx_len


def layer_weights(layer) -> tuple:
    """One layer's weights, in the order its decode step takes them.

    Qwen3 carries a per-head query and key norm, which is the difference between
    this list and Qwen2.5's -- that one has a bias on each projection instead. The
    order is stated here only, and both the single-layer arguments and the stack's
    per-layer weights are projected from it.
    """
    attention, mlp = layer.self_attn, layer.mlp
    return (
        # Taken before the context, the rope tables, the cache and the scale.
        layer.input_layernorm.weight,
        config.linear_weight(attention.q_proj),
        config.linear_weight(attention.k_proj),
        config.linear_weight(attention.v_proj),
        attention.q_norm.weight,
        attention.k_norm.weight,
        # Taken after them.
        config.linear_weight(attention.o_proj),
        layer.post_attention_layernorm.weight,
        config.linear_weight(mlp.gate_proj),
        config.linear_weight(mlp.up_proj),
        config.linear_weight(mlp.down_proj),
    )


def _build_decoder():
    from tests.models.qwen3_1_7b.decoder import build_decoder  # noqa: PLC0415

    return build_decoder()


SPEC = dense_decode.DenseDecode(
    config=config,
    layer_weights=layer_weights,
    attention_weights=6,
    build_decoder=_build_decoder,
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
    "layer_weights",
    "run_decoder_step",
]
