"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from tilefoundry.ir.core.module import Module
from tilefoundry.passes.pass_manager import PassManager, PassPoint, target_passes
from tilefoundry.passes.transforms import ConvertHIRToTIR


def finalize(
    module: Module, entry: str | None = None, *, dims: Mapping[str, int] | None = None
) -> Module:
    """Lower one checked, memory-analyzed HIR entry through ``PassManager``."""
    lowered = PassManager([*target_passes(module, PassPoint.BEFORE_LOWER), ConvertHIRToTIR(entry=entry, dims=dims)]).run(module)
    lowered = PassManager(target_passes(lowered, PassPoint.AFTER_LOWER)).run(lowered)
    return lowered if entry is None else replace(lowered, entry=entry)


__all__ = ["finalize"]
