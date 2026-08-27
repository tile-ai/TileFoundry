"""Memory-family entry points.

M8 removes the family-local residency, cache, rectangle, and allocation
structures. Scope/Access supplies the replacement in M9; these entry points
remain as loud failures while callers migrate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import AnalysisError

SELECTOR = "memory"


@dataclass(frozen=True)
class MemoryOptions:
    """How long the placement may look, and how to reproduce what it found."""

    timeout_seconds: float = 60.0
    workers: int = 1
    random_seed: int = 0


def analyze_memory(*_args: object, **_kwargs: object) -> None:
    """Fail until M9 rebuilds memory records from Scope/Access."""
    raise AnalysisError(
        "memory projection was removed in M8; M9 rebuilds it from Scope/Access"
    )


def cache_pressure(*_args: object, **_kwargs: object) -> tuple[object, ...]:
    """Fail until M9 rebuilds cache advisories from Scope/Access."""
    raise AnalysisError(
        "cache pressure was removed in M8; M9 rebuilds it from Scope/Access"
    )


__all__ = ["MemoryOptions", "SELECTOR", "analyze_memory", "cache_pressure"]
