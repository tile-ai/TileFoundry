"""Qwen2_5_1_5B decoder layer at the real shape: loads
``model/decoder_layer.py`` with ``REAL`` and re-exports its Module plus the
IR Function nodes tests inspect directly."""
from __future__ import annotations

from pathlib import Path

from tests.models.loader import load_model
from tests.models.qwen2_5_1_5b.config import REAL

_loaded = load_model(Path(__file__).parent / "model" / "decoder_layer.py", config=REAL)

Qwen2_5_1_5B = _loaded.Qwen2_5_1_5B

# IR Function nodes, not the callables ``Module.__getattr__`` returns.
decoder_layer = Qwen2_5_1_5B.lookup("decoder_layer")
input_rms_norm = Qwen2_5_1_5B.lookup("input_rms_norm")
mlp = Qwen2_5_1_5B.lookup("mlp")
self_attention = Qwen2_5_1_5B.lookup("self_attention")
tiled_mlp = Qwen2_5_1_5B.lookup("tiled_mlp")

__all__ = [
    "Qwen2_5_1_5B",
    "decoder_layer",
    "input_rms_norm",
    "mlp",
    "self_attention",
    "tiled_mlp",
]
