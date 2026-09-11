"""The topology questions every emitter asks of a module and its target.

Host module, device module and the entry signature describe three segments of
one call, so they read the answer here rather than each deciding for
themselves.
"""

from __future__ import annotations

from tilefoundry.target.facts import TopologyFacts


def coarsest_topology(module) -> str:
    """The coarsest topology level this module's program names.

    ``effective_topologies()`` is ordered coarsest first, so the first entry
    says where this program's run of levels begins; a module that names none
    is one CTA.
    """
    declared = module.effective_topologies()
    return declared[0].name if declared else "cta"


def places_of(module, target) -> tuple[str, ...]:
    """The levels *module*'s topology leaves to the host to place.

    A program names a run of levels ending at the finest one the device runs.
    A level *target* states comes from the target instance is one no device
    register answers, so its id has to arrive with the call instead.
    """
    placed = {
        level.name
        for level in target.get_facts(TopologyFacts).topologies
        if level.from_target
    }
    return tuple(
        topology.name
        for topology in module.effective_topologies()
        if topology.name in placed
    )


def places_any(module, target) -> bool:
    """Whether this program names a level whose ids the host supplies."""
    return bool(places_of(module, target))


__all__ = ["coarsest_topology", "places_any", "places_of"]
