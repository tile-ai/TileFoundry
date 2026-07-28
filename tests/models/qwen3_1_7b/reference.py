"""The executable semantics one Qwen3-1.7B decoder layer is held to.

Inputs and oracle are one pair on purpose. The oracle is a Hugging Face layer
with random weights, so a factory that returned only tensors would leave the
reference free to score them against a differently initialised layer. What
`inputs` returns therefore carries both: the arguments the evaluator is called
with, and the module those arguments were drawn from.

Everything is seeded. The same call returns the same weights and the same
activations, so a disagreement is a disagreement about the compiler rather
than about which random draw each side happened to get.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tests.models.qwen3_1_7b import config

#: Whatever runs this states its own device; the oracle is a CPU f32 baseline.
DEVICE = "cpu"

#: Seeds, named so a change to either is a visible change to the reference.
WEIGHT_SEED = 0
ACTIVATION_SEED = 1


@dataclass(frozen=True)
class DecoderLayerInputs:
    """One drawn problem: the evaluator's arguments and the layer behind them."""

    args: tuple[torch.Tensor, ...]
    layer: object
    hidden: torch.Tensor
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor
    pos_ids: torch.Tensor
    mask: torch.Tensor


def decoder_layer_inputs(*, device: str = DEVICE) -> DecoderLayerInputs:
    """One deterministic problem for the complete decoder layer."""
    layer = config.build_hf_layer(seed=WEIGHT_SEED, device=device)
    cfg = config.build_hf_config()
    sequence = config.REAL.s_cap
    cos_cache, sin_cache = config.rope_caches(cfg, sequence, device=device)
    pos_ids = torch.arange(sequence, device=device, dtype=torch.int32)
    mask = config.causal_mask(sequence, device=device)
    scale = torch.full((1, 1, 1, 1), layer.self_attn.scaling, device=device)

    torch.manual_seed(ACTIVATION_SEED)
    hidden = torch.randn(1, sequence, config.REAL.hidden, device=device) * 0.1

    attention, mlp = layer.self_attn, layer.mlp
    return DecoderLayerInputs(
        args=(
            hidden,
            layer.input_layernorm.weight,
            config.linear_weight(attention.q_proj),
            config.linear_weight(attention.k_proj),
            config.linear_weight(attention.v_proj),
            attention.q_norm.weight,
            attention.k_norm.weight,
            cos_cache,
            sin_cache,
            pos_ids,
            mask,
            scale,
            config.linear_weight(attention.o_proj),
            layer.post_attention_layernorm.weight,
            config.linear_weight(mlp.gate_proj),
            config.linear_weight(mlp.up_proj),
            config.linear_weight(mlp.down_proj),
        ),
        layer=layer,
        hidden=hidden,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        pos_ids=pos_ids,
        mask=mask,
    )


def decoder_layer_oracle(inputs: DecoderLayerInputs) -> torch.Tensor:
    """What Hugging Face's own layer produces for the same drawn problem."""
    positions = inputs.pos_ids.long()
    cos = inputs.cos_cache[positions].unsqueeze(0)
    sin = inputs.sin_cache[positions].unsqueeze(0)
    with torch.no_grad():
        return inputs.layer(
            inputs.hidden,
            position_embeddings=(cos, sin),
            attention_mask=inputs.mask,
        )


__all__ = [
    "ACTIVATION_SEED",
    "DEVICE",
    "WEIGHT_SEED",
    "DecoderLayerInputs",
    "decoder_layer_inputs",
    "decoder_layer_oracle",
]
