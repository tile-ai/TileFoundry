"""The error a malformed IR value raises."""

from __future__ import annotations

__all__ = ["VerifyError"]


class VerifyError(ValueError):
    """An IR value violates an invariant of its definition."""
