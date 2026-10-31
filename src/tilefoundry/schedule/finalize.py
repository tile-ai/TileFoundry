"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from tilefoundry.analysis import AnalysisResult
from tilefoundry.compile import default_pipeline
from tilefoundry.ir.core.module import Module


def finalize(analyzed: AnalysisResult) -> Module:
    """Lower the root of one analysis through ``PassManager``.

    *analyzed* carries at least the ``memory`` family. A Target whose lowering emits
    the synchronisation plan also needs ``sync``::

        finalize(analyze(module, entry, analysis=("memory", "sync")))
    """
    return default_pipeline(analyzed.module).run(analyzed.module)


__all__ = ["finalize"]
