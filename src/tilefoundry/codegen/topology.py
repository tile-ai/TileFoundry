"""The topology questions every emitter asks of a module and its target.

Host module, device module and the entry signature describe three segments of
one call, so they read the answer here rather than each deciding for
themselves.
"""

from __future__ import annotations


def coarsest_topology(module) -> str:
    """The coarsest topology level this module's program names.

    ``effective_topologies()`` is ordered coarsest first, so the first entry
    says where this program's run of levels begins; a module that names none
    is one CTA.
    """
    declared = module.effective_topologies()
    return declared[0].name if declared else "cta"


__all__ = ["coarsest_topology"]
