"""The published checkpoint as this model's weights: which key, and how it is stored.

The text model lives below ``model.language_model`` in this multimodal
checkpoint. Its experts are already stacked; only the fused gate/up tensor is
split while it is read.
"""
from __future__ import annotations

from functools import partial

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from tilefoundry.runtime import Absolute, Preprocessed


def _projection(t: torch.Tensor) -> torch.Tensor:
    """HF ``nn.Linear.weight`` ``(out, in)`` -> kernel ``(1, in, out)``."""
    return t.t().unsqueeze(0).contiguous()


def _transposed(t: torch.Tensor) -> torch.Tensor:
    """HF ``(out, in)`` -> an unbatched kernel ``(in, out)`` weight."""
    return t.t().contiguous()


def _gate_half(t: torch.Tensor, width: int) -> torch.Tensor:
    """The gate half of each expert's fused ``gate_up_proj`` tensor."""
    return t[:, :width, :].contiguous()


def _up_half(t: torch.Tensor, width: int) -> torch.Tensor:
    """The up half of each expert's fused ``gate_up_proj`` tensor."""
    return t[:, width:, :].contiguous()


def _squeezed(t: torch.Tensor) -> torch.Tensor:
    """HF depthwise Conv1d ``(channels, 1, kernel)`` -> ``(channels, kernel)``."""
    return t.squeeze(1).contiguous()


def hf_alias(config: Qwen3_5MoeTextConfig) -> dict[str, object]:
    """Canonical names -> published Qwen3.5 text checkpoint names for *config*."""
    width = config.moe_intermediate_size
    layers = "model.language_model.layers"
    alias: dict[str, object] = {
        "table": "model.language_model.embed_tokens.weight",
        "gamma_final": "model.language_model.norm.weight",
        "w_head": Preprocessed("lm_head.weight", _transposed),
        "head_weight_raw": "lm_head.weight",
        "gamma_in": "input_layernorm.weight",
        "gamma_q": "q_norm.weight",
        "gamma_k": "k_norm.weight",
        "gamma_post": "post_attention_layernorm.weight",
        "w_qg": Preprocessed("q_proj.weight", _projection),
        "w_k": Preprocessed("k_proj.weight", _projection),
        "w_v": Preprocessed("v_proj.weight", _projection),
        "w_o": Preprocessed("o_proj.weight", _projection),
        "w_in_qkv": Preprocessed("in_proj_qkv.weight", _projection),
        "w_in_z": Preprocessed("in_proj_z.weight", _projection),
        "w_in_b": Preprocessed("in_proj_b.weight", _projection),
        "w_in_a": Preprocessed("in_proj_a.weight", _projection),
        "conv_w": Preprocessed("conv1d.weight", _squeezed),
        "a_log": "A_log",
        "dt_bias": "dt_bias",
        "gamma_gdn": "norm.weight",
        "w_out": Preprocessed("out_proj.weight", _projection),
        "w_gate": Preprocessed(
            "experts.gate_up_proj", partial(_gate_half, width=width)
        ),
        "w_up": Preprocessed(
            "experts.gate_up_proj", partial(_up_half, width=width)
        ),
        "w_down": "experts.down_proj",
        "w_shared_gate": Preprocessed("shared_expert.gate_proj.weight", _transposed),
        "w_shared_up": Preprocessed("shared_expert.up_proj.weight", _transposed),
        "w_shared_down": Preprocessed("shared_expert.down_proj.weight", _transposed),
        "w_shared_scale": Preprocessed("shared_expert_gate.weight", _transposed),
        "w_router": Preprocessed("weight", _transposed),
    }
    for index, kind in enumerate(config.layer_types):
        layer = f"{layers}.{index}"
        mixer = "linear_attn" if kind == "linear_attention" else "self_attn"
        alias[f"layer{index}"] = layer
        alias[f"{layer}.mixer"] = mixer
        alias[f"{layer}.{mixer}.gamma_in"] = Absolute(
            f"{layer}.input_layernorm.weight"
        )
        alias[f"{layer}.moe"] = "mlp"
        alias[f"{layer}.mlp.gamma_post"] = Absolute(
            f"{layer}.post_attention_layernorm.weight"
        )
        alias[f"{layer}.mlp.router"] = "gate"
    return alias


def hf_layout_only(config: Qwen3_5MoeTextConfig) -> dict[str, Preprocessed]:
    """The table's layout entries, made relative to an in-memory test mapping."""
    return {
        name: Preprocessed(name, value.read)
        for name, value in hf_alias(config).items()
        if isinstance(value, Preprocessed)
    }


__all__ = ["hf_alias"]
