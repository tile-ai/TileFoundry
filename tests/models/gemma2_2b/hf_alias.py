"""The published checkpoint as this model's weights: which key, and how it is stored.

An entry exists where the published checkpoint and the Module disagree on a
name, stored form, or both. Gemma's zero-centred norm scales already agree with
the declarations and are read without adjustment.
"""
from __future__ import annotations

import torch
from transformers import Gemma2Config

from tilefoundry.runtime import Preprocessed


def _projection(t: torch.Tensor) -> torch.Tensor:
    """HF ``nn.Linear.weight`` ``(out, in)`` -> kernel ``(1, in, out)``."""
    return t.t().unsqueeze(0).contiguous()


def _transposed(t: torch.Tensor) -> torch.Tensor:
    """HF ``(out, in)`` -> an unbatched kernel ``(in, out)`` weight."""
    return t.t().contiguous()


def hf_alias(config: Gemma2Config) -> dict[str, object]:
    """Canonical names -> published Gemma2 checkpoint names for *config*."""
    return {
        "w_embed": "model.embed_tokens.weight",
        "gamma_final": "model.norm.weight",
        "w_head": Preprocessed("lm_head.weight", _transposed),
        "head_weight_raw": "lm_head.weight",
        "gamma_in": "input_layernorm.weight",
        "gamma_post_attn": "post_attention_layernorm.weight",
        "gamma_pre_ff": "pre_feedforward_layernorm.weight",
        "gamma_post_ff": "post_feedforward_layernorm.weight",
        "w_q": Preprocessed("self_attn.q_proj.weight", _projection),
        "w_k": Preprocessed("self_attn.k_proj.weight", _projection),
        "w_v": Preprocessed("self_attn.v_proj.weight", _projection),
        "w_o": Preprocessed("self_attn.o_proj.weight", _projection),
        "w_gate": Preprocessed("mlp.gate_proj.weight", _projection),
        "w_up": Preprocessed("mlp.up_proj.weight", _projection),
        "w_down": Preprocessed("mlp.down_proj.weight", _projection),
        **{f"layer{i}": f"model.layers.{i}" for i in range(config.num_hidden_layers)},
    }


def hf_layout_only(config: Gemma2Config) -> dict[str, Preprocessed]:
    """The table's layout entries, made relative to an in-memory test mapping."""
    return {
        name: Preprocessed(name, value.read)
        for name, value in hf_alias(config).items()
        if isinstance(value, Preprocessed)
    }


__all__ = ["hf_alias"]
