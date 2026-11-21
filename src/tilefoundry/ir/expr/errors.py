"""The error a malformed IR value raises."""

from __future__ import annotations

from .metadata import SourceSpanMetadata

__all__ = ["VerifyError"]


class VerifyError(ValueError):
    """An IR value violates an invariant of its definition; *span* is its source, when known."""

    def __init__(self, message: str, span: SourceSpanMetadata | None = None) -> None:
        super().__init__(message)
        self.span = span
