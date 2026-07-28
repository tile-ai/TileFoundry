"""Real-scale DeepSeek-V4-Flash attention: loads ``model/attention.py`` with
``REAL`` and hands back the built ``Module``."""
from __future__ import annotations

from pathlib import Path

from tests.models.deepseek_v4_flash.config import REAL
from tests.models.loader import load_model
from tilefoundry.ir.core.module import Module

_MODEL_PATH = Path(__file__).parent / "model" / "attention.py"


def build_attention() -> Module:
    """A real-scale attention Module nothing else holds a reference to.

    Re-executing the source is what makes that true: an analysis annotates the
    Call objects it measured in place, so two callers sharing one built Module
    would share each other's results.
    """
    return load_model(_MODEL_PATH, config=REAL).DeepseekV4Attention


__all__ = ["build_attention"]
