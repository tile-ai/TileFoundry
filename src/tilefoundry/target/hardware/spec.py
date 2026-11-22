"""Resolve and render the installed hardware documents behind a target."""

from __future__ import annotations

from .envelope import (
    Document,
    UnknownDocumentError,
)


def documents(target) -> tuple[Document, Document]:
    """The architecture and device documents *target* was built from.

    A target that holds no document has none to report, and says so rather
    than guessing which installed resource it resembles.
    """
    held = target.documents()
    if not held:
        raise UnknownDocumentError(
            f"target {target.identity!r} holds no installed hardware documents"
        )
    return held


def format_capabilities(
    pair: tuple[Document, Document],
) -> str:
    """Format the stable, intentionally compact capabilities report."""
    architecture, device = pair
    lines = [
        f"architecture: {architecture.id}",
        f"  digest: {architecture.digest}",
        f"device: {device.id}",
        f"  digest: {device.digest}",
        "facts:",
    ]
    for document in (architecture, device):
        for path in sorted(document.facts):
            fact = document.facts[path]
            if not fact.available:
                lines.append(f"  {path}: unavailable")
            else:
                unit = f" {fact.unit}" if fact.unit and fact.unit != "name" else ""
                lines.append(f"  {path}: {fact.value}{unit} [{fact.origin}]")
            if fact.conditions:
                lines.append(f"    conditions: {fact.conditions}")
            if fact.source:
                lines.append(f"    source: {fact.source}")
    return "\n".join(lines)


__all__ = ["documents", "format_capabilities"]
