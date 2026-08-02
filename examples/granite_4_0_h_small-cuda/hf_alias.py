"""The published checkpoint as this model's weights: which key, and how it is stored.

Almost every entry is a rename. That is on purpose: `model.py` declares each
weight in the orientation `nn.Linear` already stores it, so there is nothing to
transpose on the way in. What is left is three shape facts the checkpoint states
differently from the way a kernel wants to read them:

* the depthwise convolution is stored as a `Conv1d` weight with a singleton
  input-channel axis, which the mixer does not use;
* each layer's experts keep one fused `input_linear`, whose two halves are the
  SwiGLU gate and its multiplicand;
* the layer-level norms sit beside the block that consumes them rather than
  inside it, so they are reached with `Absolute`.
"""
from __future__ import annotations

from functools import partial

import torch
from transformers.models.granitemoehybrid.configuration_granitemoehybrid import (
    GraniteMoeHybridConfig,
)

from tilefoundry.runtime import Absolute, Preprocessed


def _squeezed(t: torch.Tensor) -> torch.Tensor:
    """HF depthwise ``Conv1d`` ``(channels, 1, kernel)`` -> ``(channels, kernel)``."""
    return t.squeeze(1).contiguous()


def _gate_half(t: torch.Tensor, width: int) -> torch.Tensor:
    """The activated half of each expert's fused ``input_linear``."""
    return t[:, :width, :].contiguous()


def _up_half(t: torch.Tensor, width: int) -> torch.Tensor:
    """The multiplicand half of each expert's fused ``input_linear``."""
    return t[:, width : 2 * width, :].contiguous()


def hf_alias(config: GraniteMoeHybridConfig) -> dict[str, object]:
    """Canonical names -> published Granite-4.0-H checkpoint names for *config*."""
    width = config.intermediate_size
    alias: dict[str, object] = {
        # root
        "table": "model.embed_tokens.weight",
        "gamma_final": "model.norm.weight",
        # mamba mixer
        "w_in": "in_proj.weight",
        "conv_w": Preprocessed("conv1d.weight", _squeezed),
        "conv_b": "conv1d.bias",
        "a_log": "A_log",
        "dt_bias": "dt_bias",
        "d_skip": "D",
        "gamma_ssm": "norm.weight",
        "w_out": "out_proj.weight",
        # full attention
        "w_q": "q_proj.weight",
        "w_k": "k_proj.weight",
        "w_v": "v_proj.weight",
        "w_o": "o_proj.weight",
        # mixture of experts
        "w_gate": Preprocessed("input_linear.weight", partial(_gate_half, width=width)),
        "w_up": Preprocessed("input_linear.weight", partial(_up_half, width=width)),
        "w_down": "output_linear.weight",
        "w_router": "layer.weight",
    }
    for index, kind in enumerate(config.layer_types):
        layer = f"model.layers.{index}"
        mixer = "mamba" if kind == "linear_attention" else "self_attn"
        alias[f"layer{index}"] = layer
        alias[f"{layer}.mixer"] = mixer
        # Both pre-norms live on the layer, one level above the block that
        # fuses them, so each is reached absolutely rather than by prefix.
        alias[f"{layer}.{mixer}.gamma_in"] = Absolute(f"{layer}.input_layernorm.weight")
        alias[f"{layer}.moe"] = "block_sparse_moe"
        alias[f"{layer}.block_sparse_moe.gamma_post"] = Absolute(
            f"{layer}.post_attention_layernorm.weight"
        )
        # The dense shared MLP is a sibling of the sparse block, not a child.
        alias[f"{layer}.block_sparse_moe.w_shared_in"] = Absolute(
            f"{layer}.shared_mlp.input_linear.weight"
        )
        alias[f"{layer}.block_sparse_moe.w_shared_out"] = Absolute(
            f"{layer}.shared_mlp.output_linear.weight"
        )
    return alias


__all__ = ["hf_alias"]
