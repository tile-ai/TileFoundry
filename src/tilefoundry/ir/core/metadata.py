from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IRMetadata:
    """Opaque metadata attached to an immutable IR expression."""


__all__ = ["IRMetadata"]
