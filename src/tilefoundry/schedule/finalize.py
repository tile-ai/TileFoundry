"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from tilefoundry.analysis import AnalysisResult
from tilefoundry.compile import default_pipeline
from tilefoundry.ir.core.module import Module


def finalize(analyzed: AnalysisResult) -> Module:
    """Lower the root of one memory analysis through ``PassManager``."""
    return default_pipeline(analyzed.module).run(analyzed.module)


__all__ = ["finalize"]
