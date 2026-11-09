"""Resolve and render the installed hardware documents behind a target."""

from __future__ import annotations

from .envelope import (
    Document,
    UnknownDocumentError,
)


def documents(target: object) -> tuple[Document, Document]:
    """The architecture and device documents *target* was composed from.

    A target holding a resource that was supplied directly rather than
    selected from the installed namespace has no document to report, and says
    so rather than guessing which installed resource it resembles.
    """
    architecture_document = getattr(target, "architecture_document", None)
    device_document = getattr(target, "device_document", None)
    if architecture_document is None or device_document is None:
        architecture = getattr(getattr(target, "architecture", None), "name", "unknown")
        device = getattr(getattr(target, "device", None), "name", "unknown")
        raise UnknownDocumentError(
            "no installed hardware documents for "
            f"device={device!r}, architecture={architecture!r}: this target was "
            "composed from directly supplied values"
        )
    return architecture_document, device_document


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
