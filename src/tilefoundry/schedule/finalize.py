"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from tilefoundry.analysis import AnalysisResult
from tilefoundry.compile import default_pipeline
from tilefoundry.ir.core.module import Module


def finalize(analyzed: AnalysisResult) -> Module:
    """Lower the root of one analysis through ``PassManager``.

    *analyzed* carries the families the Target's lowering pass names in
    ``analyses``::

        lowering = module.resolve_target().resolve(ConvertHIRToTIR, CodegenOptions())
        finalize(analyze(module, entry, analysis=lowering.analyses))
    """
    return default_pipeline(analyzed.module).run(analyzed.module)


__all__ = ["finalize"]
