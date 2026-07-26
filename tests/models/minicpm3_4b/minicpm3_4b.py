"""MiniCPM3_4B decoder layer at the real shape: loads
``model/decoder_layer.py`` with ``REAL`` and re-exports its Module plus the
IR Function nodes tests inspect directly."""
from __future__ import annotations

from pathlib import Path

from tests.models.loader import load_model
from tests.models.minicpm3_4b.config import REAL

_loaded = load_model(Path(__file__).parent / "model" / "decoder_layer.py", config=REAL)

MiniCPM3_4B = _loaded.MiniCPM3_4B

# IR Function nodes, not the callables ``Module.__getattr__`` returns.
decoder_layer = MiniCPM3_4B.lookup("decoder_layer")
input_rms_norm = MiniCPM3_4B.lookup("input_rms_norm")
mla_attention = MiniCPM3_4B.lookup("mla_attention")
mlp = MiniCPM3_4B.lookup("mlp")

__all__ = [
    "MiniCPM3_4B",
    "decoder_layer",
    "input_rms_norm",
    "mla_attention",
    "mlp",
]
