"""The `inspect` command: print the installed facts about a selection's target."""

from __future__ import annotations

import sys

from tilefoundry.cli.source import load_authored_ir, selected_target
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.target import CudaTarget
from tilefoundry.target.hardware import format_capabilities, hardware_documents


def _grid_cta_count(ir: Module | Function) -> int | None:
    if not isinstance(ir, Module):
        return None
    counts = {
        topology.size
        for topology in ir.effective_topologies()
        if topology.name == "cta" and isinstance(topology.size, int)
    }
    return next(iter(counts)) if len(counts) == 1 else None


def run_capabilities(source: str) -> int:
    """Print the capabilities of the target the selection declares."""
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
