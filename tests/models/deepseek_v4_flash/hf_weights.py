"""Real FP8 checkpoint weight loader for the DeepSeek V4 flash decode step
(M4: real-weight E2E, docs/plans/agent-kernel-loop/P0a-tonight-nested-module-e2e.md).

Reads a subset of tensors directly out of the repacked FP8 safetensors
checkpoint at ``ckpt_dir`` (46 shards + ``config.json`` +
``model.safetensors.index.json``) and remaps them to this fixture's
module-path-prefixed weight-dict keys (see ``decode_step.py`` / ``moe.py``),
without ever materializing a whole shard into memory: each key is looked up
in the index for its shard filename, and only that one tensor is read via
``safetensors.safe_open`` (an mmap'd, lazy-per-tensor read) straight onto
``cuda`` — no intermediate CPU tensor, no unrelated tensor in the same shard
(attention / hash-router "hc_*" correction terms) ever touched.

Checkpoint facts (empirically verified against a real
``DeepSeek-V4-Flash-FP8`` checkpoint on disk, not assumed):

- fp8 weights: safetensors dtype ``F8_E4M3`` -> ``torch.float8_e4m3fn``.
- block scales (routed + shared expert, 128x128 block): safetensors dtype
  is **``F32`` already** — *not* a packed ue8m0 byte code needing a
  bit-level decode. Sampled values are exact powers of two (e.g.
  ``0.00024414... == 2**-12``), consistent with the "ue8m0" *semantics*
  ``config.json``'s ``quantization_config`` declares, but the repack this
  checkpoint went through already expanded them to plain float32. Loading
  a scale is therefore a direct, lossless read — no conversion at all,
  let alone one that would need a new IR dtype.
- Router: ``config.json``'s ``num_hash_layers=3`` means layers 0..2 use a
  hash router (key ``ffn.gate.tid2eid``, no bias tensor; ``moe.py``'s
  ``moe_hash_gather``) while layers 3..42 use the learned noaux_tc router
  (``ffn.gate.weight`` + ``ffn.gate.bias``; ``moe.py``'s ``moe_topk``).
  :func:`load_decode_step_weights` loads whichever gate tensor the
  requested layer actually has. ``tid2eid`` is declared
  ``dtype=torch.int32`` in model.py but is empirically **I64** on disk in
  this checkpoint — loaded as-is (i64), which also happens to already be
  the dtype ``moe_hash_gather`` needs.
- attention weights: real transformer layer 0 (config.json
  ``compress_ratios[0] == 0``, pure sliding-window MLA — the only real
  layer ``attention.py``'s ``mla_kv_update_v2`` / ``mla_attend_v2``
  structurally implement) load via :func:`load_layer0_attention_weights`,
  which reuses ``hf_attention_ref.py``'s ``load_attention_weights`` (128x128
  block-scale FP8 -> bf16 dequant, already verified against the oracle in
  ``test_attention_hir_oracle.py``) rather than re-implementing that dequant
  a second time here, then reshapes/transposes the result to the exact
  shapes ``mla_kv_update_v2`` / ``mla_attend_v2`` expect (that file's own
  "does not import hf_weights.py" note predates this task, from when a
  different agent owned this file; both files are owned together now).
  Any other layer's attention (compress_ratio != 0: Compressor/Indexer KV
  compression) is out of scope — not expressible in HIR, see attention.py.
- every routed-expert-related key lives in exactly one shard per layer
  (verified across all 43 layers via ``model.safetensors.index.json``), so
  one ``safe_open`` handle per shard, reused for every key read out of it,
  is enough — no shard is opened twice and no other layer's shard is ever
  opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from tests.models.deepseek_v4_flash import attention as attn
from tests.models.deepseek_v4_flash.hf_attention_ref import AttnConfig, load_attention_weights
from tests.models.deepseek_v4_flash.moe import N_ROUTED

# config.json: num_hash_layers. Layers below this use hash routing
# (ffn.gate.tid2eid) instead of the learned gate.weight/gate.bias
# moe.py's moe_topk implements.
HASH_ROUTER_LAYERS = 3


class _ShardReader:
    """Opens each safetensors shard at most once (keyed by filename) and
    reads exactly the tensors asked for, straight onto ``device``."""

    def __init__(self, ckpt_dir: Path, weight_map: dict[str, str], device: str) -> None:
        self._ckpt_dir = ckpt_dir
        self._weight_map = weight_map
        self._device = device
        self._handles: dict[str, object] = {}

    def __call__(self, key: str) -> torch.Tensor:
        shard = self._weight_map[key]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self._ckpt_dir / shard), framework="pt", device=self._device)
            self._handles[shard] = handle
        return handle.get_tensor(key)


def _weight_map(ckpt_dir: Path) -> dict[str, str]:
    with open(ckpt_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    return index["weight_map"]


def _stack_experts(get: _ShardReader, layer: int, proj: str, field: str) -> torch.Tensor:
    """Stack all ``N_ROUTED`` experts' ``ffn.experts.{e}.{proj}.{field}``
    along a new leading axis — each real per-expert tensor is already
    exactly this fixture's per-expert (routed_*) shape, so stacking 256 of
    them reproduces moe.py's ``(N_ROUTED, ...)`` ConstTensor shape."""
    return torch.stack(
        [get(f"layers.{layer}.ffn.experts.{e}.{proj}.{field}") for e in range(N_ROUTED)],
        dim=0,
    )


def load_decode_step_weights(ckpt_dir: str | Path, layers: list[int]) -> dict[str, torch.Tensor]:
    """Load one real decoder layer's MoE + embed/norm/lm_head weights from
    the FP8 checkpoint at ``ckpt_dir``, keyed the same way
    ``decode_step.run_decode_step`` / ``WeightLoader`` expect
    (module-path-prefixed; see ``decode_step.py``'s module docstring).

    ``decode_step_module`` has exactly one decoder-layer slot ("layer0"), so
    ``layers`` must be a single-element list — the real checkpoint layer
    index to load into that slot. ``layer < HASH_ROUTER_LAYERS`` loads
    ``ffn.gate.tid2eid`` (for ``moe.py``'s ``moe_hash_gather``); otherwise it
    loads ``ffn.gate.bias`` (for ``moe_topk``) — ``layer0.moe.gate_weight``
    plus the routed/shared expert weights are common to both and always
    loaded, since the expert storage format does not depend on the routing
    scheme.

    Attention weights are NOT included (see module docstring) — the caller
    must merge in ``layer0.attention.*`` weights (real, via
    :func:`load_layer0_attention_weights`, or random) before handing the
    result to ``WeightLoader.load``. Fp8 weights stay
    ``torch.float8_e4m3fn``; block scales are read as plain ``torch.float32``
    (already their on-disk dtype); embed / norm / lm_head keep their
    checkpoint dtype (``bf16``) except ``layer0.moe.rms_weight``, upcast to
    f32 to match ``moe.py``'s ``pre_moe_rms_norm`` ConstTensor declaration.
    Every returned tensor is on ``cuda``.
    """
    if len(layers) != 1:
        raise NotImplementedError(
            "decode_step_module has exactly one decoder-layer slot (\"layer0\"); "
            f"pass a single real checkpoint layer index, got {layers!r}"
        )
    layer = layers[0]

    ckpt_dir = Path(ckpt_dir)
    get = _ShardReader(ckpt_dir, _weight_map(ckpt_dir), device="cuda")

    w: dict[str, torch.Tensor] = {
        "embed.table": get("embed.weight"),
        "final_rms_norm.weight": get("norm.weight"),
        "lm_head.weight": get("head.weight").t().contiguous(),
        "layer0.moe.rms_weight": get(f"layers.{layer}.ffn_norm.weight").to(torch.float32),
        "layer0.moe.gate_weight": get(f"layers.{layer}.ffn.gate.weight"),
    }
    if layer < HASH_ROUTER_LAYERS:
        w["layer0.moe.tid2eid"] = get(f"layers.{layer}.ffn.gate.tid2eid")
    else:
        w["layer0.moe.gate_bias"] = get(f"layers.{layer}.ffn.gate.bias")
    for proj in ("w1", "w3", "w2"):
        w[f"layer0.moe.routed_{proj}_weight"] = _stack_experts(get, layer, proj, "weight")
        w[f"layer0.moe.routed_{proj}_scale"] = _stack_experts(get, layer, proj, "scale")
        w[f"layer0.moe.shared_expert.{proj}_weight"] = get(f"layers.{layer}.ffn.shared_experts.{proj}.weight")
        w[f"layer0.moe.shared_expert.{proj}_scale"] = get(f"layers.{layer}.ffn.shared_experts.{proj}.scale")
    return w


def load_layer0_attention_weights(ckpt_dir: str | Path, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Load real transformer layer 0's full attention weight set (config.json
    ``compress_ratios[0] == 0`` — pure sliding-window MLA, the only real
    layer ``attention.py``'s ``mla_kv_update_v2`` / ``mla_attend_v2``
    structurally implement) and reshape it to the exact ``layer0.attention.*``
    HIR call shapes those two ``@func``s expect.

    Reuses ``hf_attention_ref.load_attention_weights`` (128x128 block-scale
    FP8 -> bf16 dequant, already cross-checked against those two ``@func``s
    in ``test_attention_hir_oracle.py`` at cosine ~0.9998) instead of
    re-implementing that dequant a second time here; this function only adds
    the transpose/reshape/view step from ``load_attention_weights``'s
    Linear-convention ``[out, in]`` weights to the HIR funcs' ``[in, out]``
    ``tf.matmul`` convention (same reshapes ``test_attention_hir_oracle.py``
    already applies inline).
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = AttnConfig.from_config_json(str(ckpt_dir / "config.json"))
    if cfg.compress_ratios[0] != 0:
        raise ValueError(
            f"layer 0 compress_ratio={cfg.compress_ratios[0]!r}, expected 0 (pure "
            "sliding-window) -- mla_kv_update_v2/mla_attend_v2 only implement that structure"
        )
    raw = load_attention_weights(0, cfg, ckpt_dir=str(ckpt_dir), device=device)
    wo_a_grouped = raw["wo_a"].view(attn.REAL_O_GROUPS, attn.REAL_O_LORA_RANK, attn.REAL_WO_A_IN)
    return {
        "layer0.attention.gamma_kv": raw["kv_norm.weight"].to(torch.bfloat16),
        "layer0.attention.w_kv": raw["wkv"].t().contiguous(),
        "layer0.attention.gamma_q_lora": raw["q_norm.weight"].to(torch.bfloat16),
        "layer0.attention.w_q_a": raw["wq_a"].t().contiguous(),
        "layer0.attention.w_q_b": raw["wq_b"].t().contiguous(),
        "layer0.attention.attn_sink": raw["attn_sink"].reshape(1, attn.REAL_N_HEADS, 1, 1).to(torch.bfloat16),
        "layer0.attention.w_o_a": wo_a_grouped.transpose(1, 2).contiguous(),
        "layer0.attention.w_o_b": raw["wo_b"].t().contiguous(),
    }


__all__ = ["HASH_ROUTER_LAYERS", "load_decode_step_weights", "load_layer0_attention_weights"]
