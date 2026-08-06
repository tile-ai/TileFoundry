"""The `inspect` command: print the installed facts about a selection's target."""

from __future__ import annotations

import sys

from tilefoundry.cli.source import load_authored_ir, selected_target
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.target import CudaTarget, registered_targets
from tilefoundry.target.hardware import HARDWARE_SPECS, format_capabilities, hardware_documents


def _grid_cta_count(ir: Module | Function) -> int | None:
    if not isinstance(ir, Module):
        return None
    counts = {
        topology.size
        for topology in ir.effective_topologies()
        if topology.name == "cta" and isinstance(topology.size, int)
    }
    return next(iter(counts)) if len(counts) == 1 else None


def _installed_capabilities() -> str:
    """Describe the hardware documents and target names available to a module."""
    documents = sorted(
        (HARDWARE_SPECS.document(spec_id) for spec_id in HARDWARE_SPECS.installed_ids()),
        key=lambda document: (document.kind, document.id),
    )
    lines = ["Installed hardware documents:"]
    for document in documents:
        compatibility = ""
        if document.compatibility:
            compatible_kind = "architectures" if document.kind == "device" else "devices"
            compatibility = f"     {compatible_kind}: {', '.join(document.compatibility)}"
        lines.append(
            f"  {document.kind:<12}  {document.id:<17}{document.schema}{compatibility}"
        )
    lines += [
        "",
        "Registered Target classes: "
        f"{', '.join(sorted(registered_targets()))}",
        "",
        "Name a SOURCE for the facts of the target that selection declares:",
        "  tilefoundry inspect capabilities model.py:Model",
    ]
    return "\n".join(lines)


def run_capabilities(source: str | None) -> int:
    """Print the capabilities of the target the selection declares."""
    if source is None:
        sys.stdout.write(_installed_capabilities() + "\n")
        return 0
    ir = load_authored_ir(source)
    target = selected_target(ir)
    if not isinstance(target, CudaTarget):
        raise ValueError(
            f"no installed authored-analysis hardware spec for target {target.name!r}"
        )
    sys.stdout.write(
        format_capabilities(
            hardware_documents(target), grid_cta_count=_grid_cta_count(ir)
        )
        + "\n"
    )
    return 0


__all__ = ["run_capabilities"]
