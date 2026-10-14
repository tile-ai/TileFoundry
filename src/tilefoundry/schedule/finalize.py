"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from tilefoundry.ir.core.module import Module
from tilefoundry.passes import PassManager
from tilefoundry.passes.transforms import ConvertHIRToTIR


def finalize(
    module: Module, entry: str | None = None, *, dims: Mapping[str, int] | None = None
) -> Module:
    """Lower one checked, memory-analyzed HIR entry through ``PassManager``."""
    lowered = PassManager([ConvertHIRToTIR(entry=entry, dims=dims)]).run(module)
    return lowered if entry is None else replace(lowered, entry=entry)


__all__ = ["finalize"]
