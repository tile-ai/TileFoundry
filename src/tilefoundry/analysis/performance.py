"""Place modeled work on exact logical participant sets, in the authored order.

Each occurrence holds every position the Mesh it was authored inside names, for
one CTA-local duration. A position runs one occurrence at a time and an
occurrence waits for what it reads, so the program's own placement says what
overlaps. Nothing is searched. Where the buffers sit belongs to ``memory``.
"""

from __future__ import annotations

from tilefoundry.ir.core import get_metadata
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shape_helpers import static_dim_value

from .errors import AnalysisError
from .facts import ParallelCapacityFacts
from .metadata import MemoryMetadata
from .visitor import AnalyzeContext
from .walk import reachable_functions

SELECTOR = "performance"


def analyze_performance(
    function: Function,
    context: AnalyzeContext,
) -> None:
    """Place every reachable Function's occurrences on a local timeline.

    The buffers this plan keeps live are held to the target's capacities by
    ``memory``, which this family depends on: a time reported here is a time for
    a program whose values have somewhere to sit.
    """
    module, target = context.module, context.target
    placement_facts = target.get_facts(ParallelCapacityFacts)
    topology = module.resolve_topology(placement_facts.topology)
    topology_extent = static_dim_value(topology.size)
    if topology_extent is None:
        raise AnalysisError(
            f"performance: topology {topology.name!r} has unresolved extent {topology.size!r}"
        )
    for fn in reachable_functions(function):
        memory = get_metadata(fn, MemoryMetadata)
        if memory is None:
            raise AnalysisError(
                f"function {fn.name!r}: performance needs the memory record this "
                "function was never given"
            )
        if memory.allocation is None:
            raise AnalysisError(
                f"function {fn.name!r}: performance reports the time of a program "
                "whose buffers were placed, and this one's machine names no level "
                "to place them against"
            )
    raise AnalysisError(
        "performance: the legacy projection was removed in M3 and is rebuilt in M4"
    )


__all__ = ["SELECTOR", "analyze_performance"]
