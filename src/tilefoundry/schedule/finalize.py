"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from collections.abc import Mapping

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.passes import PassManager
from tilefoundry.passes.transforms import ConvertHIRToTIR


def finalize(
    module: Module, entry: str | None = None, *, dims: Mapping[str, int] | None = None
) -> PrimFunction:
    """Lower one checked, memory-analyzed HIR entry through ``PassManager``."""
    lowered = PassManager([ConvertHIRToTIR(entry=entry, dims=dims)]).run(module)
    function = lowered.entry_function() if entry is None else lowered.lookup(entry)
    if not isinstance(function, PrimFunction):
        raise TypeError(f"finalize: {function.name!r} did not lower to a PrimFunction")
    return function


__all__ = ["finalize"]
