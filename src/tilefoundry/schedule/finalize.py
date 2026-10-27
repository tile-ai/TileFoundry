"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from tilefoundry.analysis import AnalysisResult
from tilefoundry.ir.core.module import Module
from tilefoundry.passes.pass_manager import PassManager, PassPoint, target_passes
from tilefoundry.passes.transforms import ConvertHIRToTIR


def finalize(analyzed: AnalysisResult) -> Module:
    """Lower the root of one memory analysis through ``PassManager``."""
    module = analyzed.module
    convert = module.resolve_target().resolve(ConvertHIRToTIR)
    lowered = PassManager([*target_passes(module, PassPoint.BEFORE_LOWER), convert]).run(module)
    lowered = PassManager(target_passes(lowered, PassPoint.AFTER_LOWER)).run(lowered)
    return lowered


__all__ = ["finalize"]
