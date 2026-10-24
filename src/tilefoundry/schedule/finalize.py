"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from tilefoundry.analysis import AnalysisResult
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.passes.pass_manager import PassManager, PassPoint, target_passes
from tilefoundry.passes.transforms import ConvertHIRToTIR


def finalize(
    analyzed: AnalysisResult, *, authored: Function
) -> Module:
    """Lower the root of one memory analysis through ``PassManager``."""
    module = analyzed.module
    lowered = PassManager([*target_passes(module, PassPoint.BEFORE_LOWER), ConvertHIRToTIR(analyzed=analyzed, authored=authored)]).run(module)
    lowered = PassManager(target_passes(lowered, PassPoint.AFTER_LOWER)).run(lowered)
    return lowered


__all__ = ["finalize"]
