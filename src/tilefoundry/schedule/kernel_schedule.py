"""``build_schedule_tree(tg) -> TileGraph`` -- construct an isl schedule
tree over a ``TileGraph`` directly from its topological statement order,
plus the band operations the resource solve needs: enumerate the bands to
decide over (:func:`schedule_bands`, :func:`band_statement`) and split one
into a tile/point band pair (:func:`tile_band`).

Nothing here solves. ``tg.units`` is already a dependence-respecting order
(``extract`` walks the SSA DAG in postorder), so sequencing the statements
in that order is legal by construction -- no affine solve to converge, and
no objective smuggled in from ``isl.schedule_constraints``, whose
dependence-distance goal is not the one this layer optimises for.
isl is left doing what it is unrivalled at: representing the tree,
transforming bands and generating code from them.

``coincident`` is written onto each band from ``tg.parallel_dims``, the
fact ``analysis.poly`` measures off ``domain`` + ``deps``.
"""
from __future__ import annotations

import isl

from tilefoundry.analysis.poly import TileGraph


class KernelScheduleError(RuntimeError):
    """A schedule tree the band operations cannot work on -- always raised
    with a message naming what was found instead."""


def _domain_sets(domain: "isl.union_set") -> dict[str, "isl.set"]:
    sets: list["isl.set"] = []
    domain.foreach_set(sets.append)
    return {s.get_tuple_name(): s for s in sets}


def _statement_schedule(s: "isl.set") -> "isl.schedule":
    """One statement's own domain under one identity band, so the band's
    members are that statement's own dimensions, in order."""
    sched = isl.schedule.from_domain(s.to_union_set())
    if not s.dim(isl.dim_type.SET):
        return sched
    identity = isl.multi_union_pw_aff.from_union_map(s.to_union_set().identity())
    return sched.insert_partial_schedule(identity)


def _mark_coincident(tree: "isl.schedule", parallel: dict[str, tuple[bool, ...]]):
    """``parallel``'s per-dimension flags written onto every band member."""

    def mark(node):
        if not isinstance(node, isl.schedule_node_band):
            return node
        flags = parallel.get(band_statement(node), ())
        for member, is_parallel in enumerate(flags[: node.n_member()]):
            if is_parallel:
                node = node.member_set_coincident(member, 1)
        return node

    return tree.get_root().map_descendant_bottom_up(mark).get_schedule()


def build_schedule_tree(tg: TileGraph) -> "isl.schedule":
    """Build a private sequence of identity bands from immutable analysis.

    The statements are not fused into one band: their ranks differ, so a
    padded shared band member would mean a different loop in each of them.
    """
    if not tg.units:
        raise KernelScheduleError("build_schedule_tree: tg.units is empty -- nothing to schedule")
    by_name = _domain_sets(tg.domain)
    missing = [unit.name for unit in tg.units if unit.name not in by_name]
    if missing:
        raise KernelScheduleError(
            f"build_schedule_tree: statements {missing} have no domain piece -- "
            "tg.units and tg.domain must come from one extract() run"
        )
    tree = _statement_schedule(by_name[tg.units[0].name])
    for unit in tg.units[1:]:
        tree = tree.sequence(_statement_schedule(by_name[unit.name]))
    return _mark_coincident(tree, tg.parallel_dims)


def schedule_bands(tree: "isl.schedule") -> tuple["isl.schedule_node_band", ...]:
    """Every band in ``tree``, in top-down order -- which for a
    :func:`build_schedule_tree` tree is ``tg.units`` order."""
    found: list["isl.schedule_node_band"] = []

    def visit(node) -> bool:
        if isinstance(node, isl.schedule_node_band):
            found.append(node)
        return True

    tree.get_root().foreach_descendant_top_down(visit)
    if not found:
        raise KernelScheduleError(f"schedule_bands: schedule tree carries no band node: {tree}")
    return tuple(found)


def band_statement(band: "isl.schedule_node_band") -> str:
    """The one statement ``band`` schedules."""
    sets: list["isl.set"] = []
    band.get_domain().foreach_set(sets.append)
    unique = sorted({s.get_tuple_name() for s in sets})
    if len(unique) != 1:
        raise KernelScheduleError(
            f"band_statement: band covers statements {unique} -- every band a "
            "sequenced schedule tree carries belongs to exactly one statement"
        )
    return unique[0]


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


def tile_bands(tree: "isl.schedule", sizes: dict[str, tuple[int, ...]]) -> "isl.schedule":
    """Every band in ``tree`` tiled by its own statement's ``sizes``.

    Tiling replaces one band with two, shifting every band below it in
    top-down order, so the walk runs bottom-up over the positions instead
    of over live nodes (an ``isl.schedule_node`` does not survive the tree
    it was taken from being rebuilt).
    """
    for position in reversed(range(len(schedule_bands(tree)))):
        band = schedule_bands(tree)[position]
        name = band_statement(band)
        if name not in sizes:
            raise KernelScheduleError(f"tile_bands: no tile size decided for statement {name!r}")
        tree = tile_band(band, sizes[name])
    return tree


__all__ = [
    "KernelScheduleError",
    "band_statement",
    "build_schedule_tree",
    "schedule_bands",
    "tile_band",
    "tile_bands",
]
