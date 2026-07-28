"""Qwen3.5-35B-A3B's full-attention token mixer at the published shape: loads
``model/full_attention.py`` with ``REAL`` and re-exports its Module plus the IR
Function nodes tests take."""
from __future__ import annotations

from pathlib import Path

from tests.models.loader import load_model
from tests.models.qwen3_5_35b_a3b.config import REAL

_loaded = load_model(
    Path(__file__).parent / "model" / "full_attention.py", config=REAL
)

Qwen3_5FullAttention = _loaded.Qwen3_5FullAttention

partial_rope = Qwen3_5FullAttention.lookup("partial_rope")
full_attention = Qwen3_5FullAttention.lookup("full_attention")

__all__ = ["Qwen3_5FullAttention", "full_attention", "partial_rope"]
