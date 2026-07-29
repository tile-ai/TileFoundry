"""Checkpoint alias table: canonical (module-path) name -> published Qwen3 key.

Written against the keys `Qwen3ForCausalLM` itself writes: the decoder stack under
`model.`, and `lm_head.weight` beside it rather than under it.
"""
from __future__ import annotations

from tests.models.qwen3_1_7b.config import Qwen3Shape


def hf_alias(shape: Qwen3Shape) -> dict[str, str]:
    """Canonical-name -> raw-checkpoint-name dict for *shape*.

    The four norms resolve to their own keys, the seven projections to the keys
    their converters read, and `layer{i}` is a subtree segment rather than a leaf.
    """
    return {
        "w_embed": "model.embed_tokens.weight",
        "gamma_final": "model.norm.weight",
        "head_weight_raw": "lm_head.weight",  # w_head's converter input
        **{f"layer{i}": f"model.layers.{i}" for i in range(shape.n_layers)},
        "gamma_in": "input_layernorm.weight",
        "gamma_post": "post_attention_layernorm.weight",
        "gamma_q": "self_attn.q_norm.weight",
        "gamma_k": "self_attn.k_norm.weight",
        "q_proj_weight": "self_attn.q_proj.weight",
        "k_proj_weight": "self_attn.k_proj.weight",
        "v_proj_weight": "self_attn.v_proj.weight",
        "o_proj_weight": "self_attn.o_proj.weight",
        "gate_proj_weight": "mlp.gate_proj.weight",
        "up_proj_weight": "mlp.up_proj.weight",
        "down_proj_weight": "mlp.down_proj.weight",
    }


__all__ = ["hf_alias"]
