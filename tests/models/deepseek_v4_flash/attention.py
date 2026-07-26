"""Real-scale DeepSeek-V4-Flash attention: loads ``model/attention.py`` with
``REAL`` and re-exports the built ``Module``."""
from __future__ import annotations

from pathlib import Path

from tests.models.deepseek_v4_flash.config import REAL
from tests.models.loader import load_model

_MODEL_PATH = Path(__file__).parent / "model" / "attention.py"

attention_module = load_model(_MODEL_PATH, config=REAL).DeepseekV4Attention

__all__ = ["attention_module"]
