"""The published checkpoint as this model's weights: which key each canonical name reads.

Every weight keeps the checkpoint's own layout ((out, in) for the projections), so
the table only renames; nothing is transposed, split or stacked.
"""
from __future__ import annotations

_LAYER_LEAF = {
    "gamma_in": "input_layernorm.weight",
    "gamma_post": "post_attention_layernorm.weight",
    "w_gate": "mlp.gate_proj.weight",
    "w_up": "mlp.up_proj.weight",
    "w_down": "mlp.down_proj.weight",
    # Gated delta net.
    "w_qkv": "linear_attn.in_proj_qkv.weight",
    "w_z": "linear_attn.in_proj_z.weight",
    "w_b": "linear_attn.in_proj_b.weight",
    "w_a": "linear_attn.in_proj_a.weight",
    "conv_w": "linear_attn.conv1d.weight",
    "a_log": "linear_attn.A_log",
    "dt_bias": "linear_attn.dt_bias",
    "gamma_gdn": "linear_attn.norm.weight",
    "w_out": "linear_attn.out_proj.weight",
    # Gated attention.
    "w_q": "self_attn.q_proj.weight",
    "w_k": "self_attn.k_proj.weight",
    "w_v": "self_attn.v_proj.weight",
    "w_o": "self_attn.o_proj.weight",
    "gamma_q": "self_attn.q_norm.weight",
    "gamma_k": "self_attn.k_norm.weight",
}


def hf_alias(config: dict) -> dict[str, str]:
    """Canonical names -> published keys, for a `SafetensorsResource(alias=...)`."""
    alias = {
        "table": "model.language_model.embed_tokens.weight",
        "gamma_final": "model.language_model.norm.weight",
        "w_head": "lm_head.weight",
        **_LAYER_LEAF,
    }
    for index in range(config["num_hidden_layers"]):
        alias[f"layer{index}"] = f"model.language_model.layers.{index}"
    return alias


__all__ = ["hf_alias"]
