"""The compile entry points: ``build``, ``compile``, ``jit`` and ``Options``."""

from __future__ import annotations

from .compilation import build, compile, normalize_to_module
from .jit_cache import jit
from .options import Options
