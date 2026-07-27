"""Resolve and render the installed hardware documents behind a target."""

from __future__ import annotations

from tilefoundry.target.hardware.envelope import (
    HardwareDocument,
    UnknownDocumentError,
)
from tilefoundry.target.hardware.registry import HARDWARE_SPECS


def hardware_documents(target: object) -> tuple[HardwareDocument, HardwareDocument]:
    """The architecture and device documents *target* was composed from.

    A target holding a resource that was supplied directly rather than
    selected from the installed namespace has no document to report, and says
    so rather than guessing which installed resource it resembles.
    """
    architecture_id = getattr(target, "architecture_id", None)
    device_id = getattr(target, "device_id", None)
    if architecture_id is None or device_id is None:
        architecture = getattr(getattr(target, "architecture", None), "name", "unknown")
        device = getattr(getattr(target, "device", None), "name", "unknown")
        raise UnknownDocumentError(
            "no installed hardware documents for "
            f"device={device!r}, architecture={architecture!r}: this target was "
            "composed from directly supplied values"
        )
    return (
        HARDWARE_SPECS.document(architecture_id),
        HARDWARE_SPECS.document(device_id),
    )


def format_capabilities(
    documents: tuple[HardwareDocument, HardwareDocument],
    *,
    grid_cta_count: int | None = None,
) -> str:
    """Format the stable, intentionally compact capabilities report."""
    architecture, device = documents
    lines = [
        f"architecture: {architecture.id}",
        f"  digest: {architecture.digest}",
        f"device: {device.id}",
        f"  digest: {device.digest}",
        f"grid_cta_count: {grid_cta_count if grid_cta_count is not None else 'unspecified'}",
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
