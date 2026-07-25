"""Checkpoint alias table: canonical (module-path) name -> real checkpoint
name, verified against ``/data2/models/DeepSeek-V4-Flash-FP8``'s
``model.safetensors.index.json`` (not guessed).

``RuntimeResource``'s alias model (``src/tilefoundry/runtime/resource.py``) is
strictly hierarchical: a value renames one path *segment* or *leaf* **within
the current scope**, joined onto the caller's already-accumulated prefix ("
``_resolve_key`` / ``_resolve_segment``). There is no wildcard/pattern
matching and no way for an alias to reach outside the current subtree (up to
a parent, or sideways to a sibling) -- whatever a hit resolves to is always
``f"{prefix}{hit}"``. Two consequences drive this table's shape:

- a bare (unqualified) entry serves every node at that *level* uniformly
  (e.g. ``"attention": "attn"`` fires the same way under every layer), but
- a per-instance name (the 43 real ``layer0..layer42`` segments; the
  per-expert routed-weight groups) cannot be expressed as one pattern and
  needs one literal entry per instance -- which is why this is a function of
  *config* (``config.n_layers`` for the former, ``config.n_routed`` for the
  latter) rather than a static module-level dict.

A dict key is the *canonical* name actually looked up at prepare/load time:
for a weight with a registered ``.converter(...)``, that is the converter's
own (raw-shaped) parameter name(s) -- e.g. ``attention.py``'s ``w_kv`` is
produced by a converter whose params are named ``wkv_weight`` / ``wkv_scale``,
so the entries here are ``"wkv_weight"`` / ``"wkv_scale"``, not ``"w_kv"``.
A weight with no converter is looked up by its own declared name directly
(e.g. ``"gamma_kv"``, ``"table"``).

The pre-FFN norm is why the strictly-hierarchical rule above has teeth: the
real checkpoint has exactly one such tensor per layer
(``layers.{i}.ffn_norm.weight``) and it is a LAYER-level tensor, a sibling of
``ffn`` -- so it is *not* reachable from ``moe``'s own subtree scope
(``layers.{i}.ffn.``) by any alias, since a hit always joins onto the
*current* prefix. The tensor's placement therefore fixes which node may own
it: ``causal_lm.py``'s ``DeepseekV4DecoderLayer`` declares
``pre_moe_norm_weight`` (aliased to ``ffn_norm.weight`` below, at the
*layer's* scope) and applies it before calling ``self.moe(...)``, and the hash
MoE component correspondingly keeps no pre-norm of its own -- so the entries
here cover every weight on the forward path exactly once.
"""
from __future__ import annotations

from tests.models.deepseek_v4_flash.config import DSV4Config

# A canonical name maps to one raw name, or (one-to-many, e.g. per-expert
# tensors) a tuple of raw names in declared order -- mirrors
# ``runtime.resource.AliasValue`` (not imported directly: this module stays
# dependency-free of the torch-importing ``runtime`` package).
AliasValue = "str | tuple[str, ...]"


def hf_alias(config: DSV4Config) -> "dict[str, AliasValue]":
    """One flat canonical-name -> raw-checkpoint-name dict for *config*.

    A function of *config* (not a module-level constant) because two groups
    of entries are per-instance, not patterns: the ``config.n_layers`` decoder
    layers (``"layer{i}": "layers.{i}"``) and the ``config.n_routed`` per-
    expert routed-weight groups (one-to-many tuples, stacked by ``prepare``
    along a new leading axis -- docs/spec/runtime.md §1.5 / ``resource.py``).
    """
    alias: "dict[str, AliasValue]" = {
        # ── root (DeepseekV4ForCausalLM) ────────────────────────────────
        "table": "embed.weight",
        "final_norm_weight": "norm.weight",
        "head_weight_raw": "head.weight",  # lm_head_weight's converter input
        # ── decoder-layer segment addressing (one entry per real layer;
        # no wildcard exists -- see module docstring) ──────────────────
        **{f"layer{i}": f"layers.{i}" for i in range(config.n_layers)},
        # ── decoder-layer leaves (DeepseekV4DecoderLayer's own norms) ───
        "pre_attn_norm_weight": "attn_norm.weight",
        "pre_moe_norm_weight": "ffn_norm.weight",
        # ── decoder-layer child segment addressing ──────────────────────
        "attention": "attn",
        "moe": "ffn",
        # ── attention leaves (attention.py's converter inputs / direct
        # weights) ───────────────────────────────────────────────────────
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
        # ── moe leaves (moe.py's converter inputs / direct weights) ─────
        "gate_weight": "gate.weight",
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
        # NOTE: moe.py's own internal `rms_weight` (pre_moe_rms_norm) is
        # intentionally absent -- see the module docstring above.
    }
    return alias


__all__ = ["hf_alias"]
