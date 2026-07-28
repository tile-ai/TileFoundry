"""The complete Gemma-2-2B decoder at the real shape: loads ``model/decoder.py``
with ``REAL`` and the ordered per-layer Modules, and re-exports the tree.

Each layer is a separate load of ``model/decoder_layer.py``, so the stack holds
``n_layers`` distinct Modules rather than one Module referenced 26 times. That
matters because a Module is the execution domain of the functions it owns: one
shared instance would make every layer's analysis and schedule the same object's,
and a per-layer result would have nowhere to live.
"""
from __future__ import annotations

from pathlib import Path

from tests.models.gemma2_2b.config import REAL
from tests.models.loader import load_model

_MODEL_DIR = Path(__file__).parent / "model"


def _layer(index: int):
    """The decode layer Module for position *index* in the stack."""
    loaded = load_model(_MODEL_DIR / "decoder_layer.py", config=REAL)
    return loaded.Gemma2_2B.renamed(f"layer{index}")


def build_decoder(shape=REAL):
    """The whole decoder tree, layers built in order."""
    layers = tuple(_layer(index) for index in range(shape.n_layers))
    loaded = load_model(_MODEL_DIR / "decoder.py", config=shape, decoder_layers=layers)
    return loaded.Gemma2_2B_Decoder


__all__ = ["build_decoder"]
