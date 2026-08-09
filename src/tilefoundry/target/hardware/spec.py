"""Resolve and render the installed hardware documents behind a target."""

from __future__ import annotations

from tilefoundry.target.hardware.envelope import (
    HardwareDocument,
    UnknownDocumentError,
)


def hardware_documents(target: object) -> tuple[HardwareDocument, HardwareDocument]:
    """The architecture and device documents *target* was composed from.

    A target holding a resource that was supplied directly rather than
    selected from the installed namespace has no document to report, and says
    so rather than guessing which installed resource it resembles.
    """
    architecture_document = getattr(target, "_architecture_document", None)
    device_document = getattr(target, "_device_document", None)
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
    documents: tuple[HardwareDocument, HardwareDocument],
) -> str:
    """Format the stable, intentionally compact capabilities report."""
    architecture, device = documents
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


__all__ = ["format_capabilities", "hardware_documents"]
