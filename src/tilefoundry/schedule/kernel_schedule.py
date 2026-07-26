"""``compute_schedule(tg) -> TileGraph`` -- pure-isl affine scheduling
over a ``TileGraph``'s dependences. V1 is affine-structure only: no CP-SAT
resource/tile-size decisions here (``solve_resources.py`` adds those). Named for
isl's own ``compute_schedule`` verb, so it never reads as the ``Schedule``
service this package also exports.
"""
from __future__ import annotations

import dataclasses

import isl

from tilefoundry.analysis.poly import TileGraph


def compute_schedule(tg: TileGraph) -> TileGraph:
    """Compute an isl schedule tree from ``tg``'s domain + auto-inferred
    deps (``tg.deps`` seeds validity, proximity and coincidence alike) and
    return ``tg`` with ``tree`` filled in."""
    sc = (
        isl.schedule_constraints.on_domain(tg.domain)
        .set_validity(tg.deps)
        .set_proximity(tg.deps)
        .set_coincidence(tg.deps)
    )
    return dataclasses.replace(tg, tree=sc.compute_schedule())


__all__ = ["compute_schedule"]
