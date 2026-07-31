"""Checkpoint alias table: canonical (module-path) name -> real checkpoint
name, per ``DeepSeek-V4-Flash-FP8``'s ``model.safetensors.index.json``."""
from __future__ import annotations

from tests.models.deepseek_v4_flash.model import DSV4Config
from tilefoundry.runtime import Absolute

# One raw name, one absolute raw name, or (per-expert groups) a tuple of raw
# names in declared order.
AliasValue = "str | tuple[str, ...] | Absolute"


def hf_alias(config: DSV4Config) -> "dict[str, AliasValue]":
    """Canonical-name -> raw-checkpoint-name dict for *config* (per-layer and
    per-expert entries need ``config.n_layers`` / ``config.n_routed``)."""
    alias: "dict[str, AliasValue]" = {
        "table": "embed.weight",
        "final_norm_weight": "norm.weight",
        "head_weight_raw": "head.weight",  # lm_head_weight's converter input
        **{f"layer{i}": f"layers.{i}" for i in range(config.n_layers)},
        "pre_attn_norm_weight": "attn_norm.weight",
        "pre_moe_norm_weight": "ffn_norm.weight",
        "attention": "attn",
        "moe": "ffn",
        "gamma_kv": "kv_norm.weight",
        "gamma_q_lora": "q_norm.weight",
        "wkv_weight": "wkv.weight",
        "wkv_scale": "wkv.scale",
        "wq_a_weight": "wq_a.weight",
        "wq_a_scale": "wq_a.scale",
        "wq_b_weight": "wq_b.weight",
        "wq_b_scale": "wq_b.scale",
        "attn_sink_raw": "attn_sink",
        "wo_a_weight": "wo_a.weight",
        "wo_b_weight": "wo_b.weight",
        "wo_b_scale": "wo_b.scale",
        # The router's table is stored one level up, beside the layer's norms
        # rather than inside the ffn: an absolute key is how the child reaches it.
        **{
            f"layers.{i}.ffn.gate_weight": Absolute(f"layers.{i}.gate.weight")
            for i in range(config.n_layers)
        },
        "tid2eid": "gate.tid2eid",
        "w1_weight": tuple(f"experts.{i}.w1.weight" for i in range(config.n_routed)),
        "w3_weight": tuple(f"experts.{i}.w3.weight" for i in range(config.n_routed)),
        "w2_weight": tuple(f"experts.{i}.w2.weight" for i in range(config.n_routed)),
        "w1_scale_raw": tuple(f"experts.{i}.w1.scale" for i in range(config.n_routed)),
        "w3_scale_raw": tuple(f"experts.{i}.w3.scale" for i in range(config.n_routed)),
        "w2_scale_raw": tuple(f"experts.{i}.w2.scale" for i in range(config.n_routed)),
        "shared_w1_weight": "shared_experts.w1.weight",
        "shared_w1_scale_raw": "shared_experts.w1.scale",
        "shared_w3_weight": "shared_experts.w3.weight",
        "shared_w3_scale_raw": "shared_experts.w3.scale",
        "shared_w2_weight": "shared_experts.w2.weight",
        "shared_w2_scale_raw": "shared_experts.w2.scale",
    }
    return alias


__all__ = ["hf_alias"]
