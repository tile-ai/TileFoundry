"""Public scheduled-HIR finalization entry point."""

from __future__ import annotations

from tilefoundry import analysis, passes
from tilefoundry.ir import core


def finalize(analyzed: analysis.Result) -> core.Module:
    """Lower the root of one analysis through ``PassManager``.

    *analyzed* carries the families the Target's lowering pass names in
    ``analyses``::

        lowering = module.resolve_target().resolve(ConvertHIRToTIR, passes.Options())
        finalize(analyze(module, entry, analysis=lowering.analyses))
    """
    return passes.default_pipeline(analyzed.module).run(analyzed.module)


__all__ = ["finalize"]
