"""The published checkpoint as this model's weights: which key, and how it is stored.

An entry exists where the published checkpoint and the Module disagree on a
name, stored form, or both. The MLA a/b projections are separate checkpoint
tensors, so each one keeps its own published key.
"""
from __future__ import annotations

import torch
from transformers import MiniCPM3Config

from tilefoundry.runtime import Preprocessed


def _projection(t: torch.Tensor) -> torch.Tensor:
    """HF ``nn.Linear.weight`` ``(out, in)`` -> kernel ``(1, in, out)``."""
    return t.t().unsqueeze(0).contiguous()


def _transposed(t: torch.Tensor) -> torch.Tensor:
    """HF ``(out, in)`` -> an unbatched kernel ``(in, out)`` weight."""
    return t.t().contiguous()


def hf_alias(config: MiniCPM3Config) -> dict[str, object]:
    """Canonical names -> published MiniCPM3 checkpoint names for *config*."""
    return {
        "w_embed": "model.embed_tokens.weight",
        "gamma_final": "model.norm.weight",
        "w_head": Preprocessed("lm_head.weight", _transposed),
        "head_weight_raw": "lm_head.weight",
        "gamma_in": "input_layernorm.weight",
        "gamma_q_a": "self_attn.q_a_layernorm.weight",
        "gamma_kv_a": "self_attn.kv_a_layernorm.weight",
        "gamma_post": "post_attention_layernorm.weight",
        "w_q_a": Preprocessed("self_attn.q_a_proj.weight", _projection),
        "w_q_b": Preprocessed("self_attn.q_b_proj.weight", _projection),
        "w_kv_a": Preprocessed("self_attn.kv_a_proj_with_mqa.weight", _projection),
        "w_kv_b": Preprocessed("self_attn.kv_b_proj.weight", _projection),
        "w_o": Preprocessed("self_attn.o_proj.weight", _projection),
        "w_gate": Preprocessed("mlp.gate_proj.weight", _projection),
        "w_up": Preprocessed("mlp.up_proj.weight", _projection),
        "w_down": Preprocessed("mlp.down_proj.weight", _projection),
        **{f"layer{i}": f"model.layers.{i}" for i in range(config.num_hidden_layers)},
    }


def hf_layout_only(config: MiniCPM3Config) -> dict[str, Preprocessed]:
    """The table's layout entries, made relative to an in-memory test mapping."""
    return {
        name: Preprocessed(name, value.read)
        for name, value in hf_alias(config).items()
        if isinstance(value, Preprocessed)
    }


__all__ = ["hf_alias"]
