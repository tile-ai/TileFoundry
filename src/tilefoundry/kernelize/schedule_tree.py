"""``schedule(tg) -> ScheduleTree`` -- pure-isl affine scheduling over a
``TileGraph``'s dependences. V1 is affine-structure only: no CP-SAT
resource/tile-size decisions here (that is explicitly future work --
see ``ring`` below); this is a straight port of the validated
``.tmp/poc/09_schedule_tree.py`` technique to a ``TileGraph``'s
``domain``/``deps``.
"""
from __future__ import annotations

from dataclasses import dataclass

import isl

from .tile_graph import TileGraph


@dataclass(frozen=True)
class ScheduleTree:
    """An isl schedule tree computed for one ``TileGraph``.

    ``ring`` is reserved for a later buffer-@-ring-stage annotation pass
    (the CP-SAT resource/software-pipelining decisions this module
    explicitly defers); V1 always leaves it empty.
    """

    tree: "isl.schedule"
    ring: dict

    def __str__(self) -> str:
        return str(self.tree)


def schedule(tg: TileGraph) -> ScheduleTree:
    """Compute an isl schedule tree from ``tg``'s domain + auto-inferred
    deps: ``tg.deps`` seeds validity, proximity (fusion/locality), and
    coincidence (parallel-dim hinting) alike, exactly as in PoC 09."""
    sc = (
        isl.schedule_constraints.on_domain(tg.domain)
        .set_validity(tg.deps)
        .set_proximity(tg.deps)
        .set_coincidence(tg.deps)
    )
    return ScheduleTree(tree=sc.compute_schedule(), ring={})


__all__ = ["ScheduleTree", "schedule"]
