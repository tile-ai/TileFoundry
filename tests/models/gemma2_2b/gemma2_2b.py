"""Gemma2_2B decoder layer at the real shape: loads
``model/decoder_layer.py`` with ``REAL`` and re-exports its Module plus the
IR Function nodes tests inspect directly."""
from __future__ import annotations

from pathlib import Path

from tests.models.gemma2_2b.config import REAL
from tests.models.loader import load_model

_loaded = load_model(Path(__file__).parent / "model" / "decoder_layer.py", config=REAL)

Gemma2_2B = _loaded.Gemma2_2B

# IR Function nodes, not the callables ``Module.__getattr__`` returns.
decoder_layer = Gemma2_2B.lookup("decoder_layer")
input_rms_norm = Gemma2_2B.lookup("input_rms_norm")
mlp = Gemma2_2B.lookup("mlp")
self_attention = Gemma2_2B.lookup("self_attention")

__all__ = [
    "Gemma2_2B",
    "decoder_layer",
    "input_rms_norm",
    "mlp",
    "self_attention",
]
