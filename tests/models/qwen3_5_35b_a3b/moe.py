"""Qwen3.5-35B-A3B's MoE block at the published shape: loads ``model/moe.py``
with ``REAL`` and re-exports its Module plus the IR Function nodes tests take."""
from __future__ import annotations

from pathlib import Path

from tests.models.loader import load_model
from tests.models.qwen3_5_35b_a3b.config import REAL

_loaded = load_model(Path(__file__).parent / "model" / "moe.py", config=REAL)

Qwen3_5MoE = _loaded.Qwen3_5MoE

routing = Qwen3_5MoE.lookup("routing")
routed_experts = Qwen3_5MoE.lookup("routed_experts")
shared_expert = Qwen3_5MoE.lookup("shared_expert")
moe = Qwen3_5MoE.lookup("moe")

__all__ = [
    "Qwen3_5MoE",
    "moe",
    "routed_experts",
    "routing",
    "shared_expert",
]
