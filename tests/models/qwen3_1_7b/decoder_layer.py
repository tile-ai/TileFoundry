"""Qwen3-1.7B decoder layer at the real shape: loads ``model/decoder_layer.py``
with ``REAL`` and re-exports its Module plus the IR Function nodes the schedule
and analysis tests inspect directly."""
from __future__ import annotations

from pathlib import Path

from tests.models.loader import load_model
from tests.models.qwen3_1_7b.config import REAL

_loaded = load_model(Path(__file__).parent / "model" / "decoder_layer.py", config=REAL)

Qwen3_1_7B = _loaded.Qwen3_1_7B

# Block shape of the loop-tiled MLP, derived from the AMX register files by the
# model source; re-exported for tests that assert against the numbers.
MT, NT, KT = _loaded.MT, _loaded.NT, _loaded.KT
MB, NB_INT, NB_HID = _loaded.MB, _loaded.NB_INT, _loaded.NB_HID
NK_HID, NK_INT = _loaded.NK_HID, _loaded.NK_INT

# IR Function nodes, not the callables ``Module.__getattr__`` returns: these are
# what extract / build_schedule_tree take.
input_rms_norm = Qwen3_1_7B.lookup("input_rms_norm")
self_attention = Qwen3_1_7B.lookup("self_attention")
mlp = Qwen3_1_7B.lookup("mlp")
tiled_mlp = Qwen3_1_7B.lookup("tiled_mlp")
decoder_layer = Qwen3_1_7B.lookup("decoder_layer")

__all__ = [
    "KT",
    "MB",
    "MT",
    "NB_HID",
    "NB_INT",
    "NK_HID",
    "NK_INT",
    "NT",
    "Qwen3_1_7B",
    "decoder_layer",
    "input_rms_norm",
    "mlp",
    "self_attention",
    "tiled_mlp",
]
