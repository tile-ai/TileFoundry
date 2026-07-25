"""``schedule(tg) -> TileGraph`` -- pure-isl affine scheduling over a
``TileGraph``'s dependences. V1 is affine-structure only: no CP-SAT
resource/tile-size decisions here (``solve_resources.py`` adds those);
this is a straight port of the validated ``.tmp/poc/09_schedule_tree.py``
technique to a ``TileGraph``'s ``domain``/``deps``.
"""
from __future__ import annotations

import dataclasses

import isl

from .tile_graph import TileGraph


def schedule(tg: TileGraph) -> TileGraph:
    """Compute an isl schedule tree from ``tg``'s domain + auto-inferred
    deps (``tg.deps`` seeds validity, proximity and coincidence alike,
    exactly as in PoC 09) and return ``tg`` with ``tree`` filled in."""
    sc = (
        isl.schedule_constraints.on_domain(tg.domain)
        .set_validity(tg.deps)
        .set_proximity(tg.deps)
        .set_coincidence(tg.deps)
    )
    return dataclasses.replace(tg, tree=sc.compute_schedule())


__all__ = ["schedule"]
