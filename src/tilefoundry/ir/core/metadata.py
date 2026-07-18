from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeVar

T = TypeVar("T", bound="IRMetadata")


@dataclass(frozen=True)
class IRMetadata:
    """Base class for typed metadata attached to an IR expression."""

    def format_comment(self) -> str | None:
        """Return an optional source-printer comment for this metadata."""
        return None


def get_metadata(expr: "Expr", cls: type[T]) -> T | None:
    """Return the metadata whose concrete class is ``cls``, if present."""
    return next((value for value in expr.metadata if type(value) is cls), None)


def replace_metadata(expr: "Expr", value: IRMetadata) -> "Expr":
    """Return ``expr`` with metadata of ``value``'s concrete class replaced."""
    value_cls = type(value)
    found = False
    updated = []
    for current in expr.metadata:
        if type(current) is value_cls:
            updated.append(value)
            found = True
        else:
            updated.append(current)
    if not found:
        updated.append(value)
    return replace(expr, metadata=tuple(updated))


def remove_metadata(expr: "Expr", cls: type[IRMetadata]) -> "Expr":
    """Return ``expr`` without metadata whose concrete class is ``cls``."""
    updated = tuple(value for value in expr.metadata if type(value) is not cls)
    if updated == expr.metadata:
        return expr
    return replace(expr, metadata=updated)


__all__ = ["IRMetadata", "get_metadata", "remove_metadata", "replace_metadata"]
