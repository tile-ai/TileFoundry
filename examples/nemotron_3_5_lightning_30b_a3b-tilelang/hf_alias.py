"""The published checkpoint as this model's weights: which key, and how it is stored.

The whole decode step is one `@func` on one Module, so every weight is named at
the root and this table is flat. Three kinds of entry appear:

- a plain rename, where the checkpoint already stores exactly what the step
  declares. That is nearly all of them: every projection is declared `(out, in)`,
  which is the layout `nn.Linear.weight` has, and the step reads it with
  ``b_layout="NK"``;
- a `Preprocessed` rename for the depthwise convolution, whose published weight
  carries a singleton input-channel axis this model has no use for;
- a tuple, for the routed experts: the checkpoint ships one tensor per expert and
  the step declares one stacked tensor, so `prepare` assembles the 128 with
  `torch.stack`, the one reshaping `prepare` does.
"""
from __future__ import annotations

import torch

from tilefoundry.runtime import Preprocessed


def _squeeze1(x: torch.Tensor) -> torch.Tensor:
    """HF depthwise ``Conv1d.weight`` ``(channels, 1, kernel)`` -> ``(channels, kernel)``."""
    return x.squeeze(1).contiguous()


def hf_alias(config: dict) -> dict[str, object]:
    """Canonical weight names -> published checkpoint names for *config*."""
    kinds = [{"mamba": "linear_attention", "attention": "full_attention"}.get(k, k)
             for k in config["layers_block_type"]]
    experts = config["n_routed_experts"]
    alias: dict[str, object] = {
        "table": "backbone.embeddings.weight",
        "gamma_final": "backbone.norm_f.weight",
        "w_head": "lm_head.weight",
    }
    for index, kind in enumerate(kinds):
        mixer = f"backbone.layers.{index}.mixer"
        alias[f"l{index}_gamma"] = f"backbone.layers.{index}.norm.weight"
        if kind == "linear_attention":
            alias[f"l{index}_w_in"] = f"{mixer}.in_proj.weight"
            alias[f"l{index}_conv_w"] = Preprocessed(f"{mixer}.conv1d.weight", _squeeze1)
            alias[f"l{index}_conv_b"] = f"{mixer}.conv1d.bias"
            alias[f"l{index}_a_log"] = f"{mixer}.A_log"
            alias[f"l{index}_dt_bias"] = f"{mixer}.dt_bias"
            alias[f"l{index}_d_skip"] = f"{mixer}.D"
            alias[f"l{index}_gamma_gdn"] = f"{mixer}.norm.weight"
            alias[f"l{index}_w_out"] = f"{mixer}.out_proj.weight"
        elif kind == "full_attention":
            alias[f"l{index}_w_q"] = f"{mixer}.q_proj.weight"
            alias[f"l{index}_w_k"] = f"{mixer}.k_proj.weight"
            alias[f"l{index}_w_v"] = f"{mixer}.v_proj.weight"
            alias[f"l{index}_w_o"] = f"{mixer}.o_proj.weight"
        else:
            alias[f"l{index}_w_router"] = f"{mixer}.gate.weight"
            alias[f"l{index}_e_bias"] = f"{mixer}.gate.e_score_correction_bias"
            alias[f"l{index}_w_up"] = tuple(
                f"{mixer}.experts.{e}.up_proj.weight" for e in range(experts)
            )
            alias[f"l{index}_w_down"] = tuple(
                f"{mixer}.experts.{e}.down_proj.weight" for e in range(experts)
            )
            alias[f"l{index}_w_sh_up"] = f"{mixer}.shared_experts.up_proj.weight"
            alias[f"l{index}_w_sh_down"] = f"{mixer}.shared_experts.down_proj.weight"
    return alias


__all__ = ["hf_alias"]
