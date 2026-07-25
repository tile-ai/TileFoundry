"""DeepSeek-V4-Flash dimensions, read from the model's own ``config.json``.

The checked-in ``config.json`` is the real one (``DeepSeek-V4-Flash-FP8``), so
every dimension below is the model's, not a transcription. ``transformers`` is
deliberately not used: ``deepseek_v4`` only exists in 5.x while CI pins
``transformers>=4.57``, and the checkpoint directory is not available in CI.

``REAL`` is the shape the structural / schedule fixtures use. ``tiny()`` is the
same model at a size an end-to-end numeric test can run; its dimensions are
chosen to keep every divisibility the real code path relies on (see ``tiny``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Sourced from the modeling code, not config.json: the KV-cache fp8 fake-quant
# block and the e4m3 grid it quantizes onto (hf_attention_ref._fake_quant_fp8_block).
KV_QUANT_BLOCK = 64
FP8E4M3_MAX = 448.0
FP8E4M3_QUANT_EPS = 1e-4  # amax floor, guards log2(0) on an all-zero block


@dataclass(frozen=True)
class DSV4Config:
    """One decoder layer's shape, plus the model-wide embedding/head shape."""

    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_dim: int
    q_lora_rank: int
    o_groups: int
    o_lora_rank: int
    window: int
    vocab: int
    moe_inter: int
    n_routed: int
    n_act: int
    route_scale: float
    swiglu_limit: float
    n_layers: int
    n_hash_layers: int
    rms_eps: float
    quant_block: int
    compress_ratios: tuple[int, ...]

    @classmethod
    def from_config_json(cls, path: str | Path) -> "DSV4Config":
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        block = cfg["quantization_config"]["weight_block_size"]
        if block[0] != block[1]:
            raise ValueError(f"non-square weight_block_size {block} is not supported")
        return cls(
            dim=cfg["hidden_size"],
            n_heads=cfg["num_attention_heads"],
            n_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            rope_dim=cfg["qk_rope_head_dim"],
            q_lora_rank=cfg["q_lora_rank"],
            o_groups=cfg["o_groups"],
            o_lora_rank=cfg["o_lora_rank"],
            window=cfg["sliding_window"],
            vocab=cfg["vocab_size"],
            moe_inter=cfg["moe_intermediate_size"],
            n_routed=cfg["n_routed_experts"],
            n_act=cfg["num_experts_per_tok"],
            route_scale=cfg["routed_scaling_factor"],
            swiglu_limit=cfg["swiglu_limit"],
            n_layers=cfg["num_hidden_layers"],
            n_hash_layers=cfg["num_hash_layers"],
            rms_eps=cfg["rms_norm_eps"],
            quant_block=block[0],
            compress_ratios=tuple(cfg["compress_ratios"]),
        )

    @classmethod
    def tiny(cls) -> "DSV4Config":
        """The same model, small enough to run end to end.

        Every dimension a shape rule divides is kept divisible: ``dim`` /
        ``moe_inter`` / ``q_lora_rank`` / ``q_proj`` / ``wo_a_out`` stay
        multiples of ``quant_block`` (so a block scale really is 2x2 rather
        than degenerating to 1x1), and ``nope_dim`` (192) stays a multiple of
        ``KV_QUANT_BLOCK`` (64). One layer, hash-routed like real layer 0.
        """
        return cls(
            dim=256,
            n_heads=2,
            n_kv_heads=1,
            head_dim=256,
            rope_dim=64,
            q_lora_rank=128,
            o_groups=2,
            o_lora_rank=128,
            window=8,
            vocab=64,
            moe_inter=256,
            n_routed=4,
            n_act=2,
            route_scale=1.5,
            swiglu_limit=10.0,
            n_layers=1,
            n_hash_layers=1,
            rms_eps=1e-6,
            quant_block=128,
            compress_ratios=(0,),
        )

    # ── derived shapes ───────────────────────────────────────────────────────

    @property
    def rope_half(self) -> int:
        return self.rope_dim // 2

    @property
    def nope_dim(self) -> int:
        """Head dims left unrotated, and the only part the KV cache quantizes."""
        return self.head_dim - self.rope_dim

    @property
    def kv_quant_blocks(self) -> int:
        return self.nope_dim // KV_QUANT_BLOCK

    @property
    def q_proj(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def wo_a_in(self) -> int:
        return self.q_proj // self.o_groups

    @property
    def wo_a_out(self) -> int:
        return self.o_groups * self.o_lora_rank

    def blocks(self, extent: int) -> int:
        """Block-scale count along an axis of *extent* (weight_block_size)."""
        if extent % self.quant_block:
            raise ValueError(
                f"extent {extent} is not a multiple of quant_block {self.quant_block}"
            )
        return extent // self.quant_block


REAL = DSV4Config.from_config_json(Path(__file__).with_name("config.json"))

__all__ = [
    "FP8E4M3_MAX",
    "FP8E4M3_QUANT_EPS",
    "KV_QUANT_BLOCK",
    "REAL",
    "DSV4Config",
]
