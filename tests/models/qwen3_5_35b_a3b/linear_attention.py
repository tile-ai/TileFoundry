"""Qwen3.5-35B-A3B's Gated DeltaNet token mixer at the published shape: loads
``model/linear_attention.py`` with ``REAL`` and re-exports its Module plus the IR
Function nodes tests take."""
from __future__ import annotations

from pathlib import Path

from tests.models.loader import load_model
from tests.models.qwen3_5_35b_a3b.config import REAL

_loaded = load_model(
    Path(__file__).parent / "model" / "linear_attention.py", config=REAL
)

Qwen3_5LinearAttention = _loaded.Qwen3_5LinearAttention

conv_step = Qwen3_5LinearAttention.lookup("conv_step")
delta_step = Qwen3_5LinearAttention.lookup("delta_step")
l2_normalise = Qwen3_5LinearAttention.lookup("l2_normalise")
linear_attention = Qwen3_5LinearAttention.lookup("linear_attention")

__all__ = [
    "Qwen3_5LinearAttention",
    "conv_step",
    "delta_step",
    "l2_normalise",
    "linear_attention",
]
