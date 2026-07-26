"""``compute_schedule(tg) -> TileGraph`` -- pure-isl affine scheduling
over a ``TileGraph``'s dependences, plus the two band operations the
resource solve needs: find the band to decide over
(:func:`outermost_band`) and split it into a tile/point band pair
(:func:`tile_band`). Named for isl's own ``compute_schedule`` verb, so it
never reads as the ``Schedule`` service this package also exports.
"""
from __future__ import annotations

import dataclasses

import isl

from tilefoundry.analysis.poly import TileGraph


class KernelScheduleError(RuntimeError):
    """A schedule tree the band operations cannot work on -- always raised
    with a message naming what was found instead."""


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


def outermost_band(tree: "isl.schedule") -> "isl.schedule_node_band":
    """The first band in top-down order -- the one whose members carry the
    ``coincident`` marks and whose extent the resource solve tiles."""
    found: list["isl.schedule_node_band"] = []

    def visit(node) -> bool:
        if not found and isinstance(node, isl.schedule_node_band):
            found.append(node)
        return not found

    tree.get_root().foreach_descendant_top_down(visit)
    if not found:
        raise KernelScheduleError(
            f"outermost_band: schedule tree carries no band node: {tree}"
        )
    return found[0]


def tile_band(band: "isl.schedule_node_band", sizes: tuple[int, ...]) -> "isl.schedule":
    """``band`` split into a tile band over ``sizes`` plus a point band
    holding the remainder, returned as the whole schedule."""
    if band.n_member() != len(sizes):
        raise KernelScheduleError(
            f"tile_band: band has {band.n_member()} member(s) but got "
            f"{len(sizes)} tile size(s)"
        )
    space = band.get_partial_schedule().get_space()
    multi = isl.multi_val.zero(space)
    for i, size in enumerate(sizes):
        if size < 1:
            raise KernelScheduleError(f"tile_band: tile size {size} at member {i} must be >= 1")
        multi = multi.set_at(i, isl.val(size))
    return band.tile(multi).get_schedule()


__all__ = ["KernelScheduleError", "compute_schedule", "outermost_band", "tile_band"]
