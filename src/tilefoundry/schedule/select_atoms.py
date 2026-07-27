"""``select_atoms(tg, target, stage) -> TileGraph`` -- pick each statement's
compute atom, and measure what that pick implies.

Every operation carries its own extent and that extent *is* its tile: what
the author wrote is what one hole computes. The one choice left per statement
is which of its candidate atoms granularises it; everything else is measured
off ``tg`` -- the ring depth from the dependence distance isl reports, the
footprint from the access relations, the nominal timeline from the picked
atom's own roofline estimate. Returns ``tg`` with the tiled tree, ``ring``
and ``decisions``.

``build_schedule_tree`` gives each statement its own identity band, so a
band member *is* a domain dimension of the one statement under it -- every
fact below is per statement, per own dimension, with no
band-member-to-domain-dimension mapping in between.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import isl

from tilefoundry.analysis.poly import (
    AccessFootprint,
    AxisExtent,
    TileGraph,
    TileUnit,
    access_footprints,
    carried_distances,
    statement_time_dims,
    time_extents,
)
from tilefoundry.target import Target, default_target, resolve_target
from tilefoundry.target.facts import TARGET_FACTS, TargetFactsError

from .facts import AtomCandidateFacts, AtomCandidateQuery, AtomFact, TileStoreFacts
from .kernel_schedule import band_statement, schedule_bands, tile_bands

# ns -> integer duration units. One atom's roofline estimate is floored at
# 1.0ns and a statement's nominal time is a sum over its instances, so a bare
# round(ns) would quantise every atom to the same "1".
_DURATION_SCALE = 1000

# Nominal per-domain-element cost (ns) for a statement with no registered
# atom candidate, whose only "how much work" signal is its own extent.
_DEFAULT_DURATION_NS = 1.0
_DEFAULT_UNITS = round(_DEFAULT_DURATION_NS * _DURATION_SCALE)


class AtomSelectionError(RuntimeError):
    """A ``tg`` consistency precondition did not hold, or the stage exposes
    no fact the decisions stand on -- always raised with a specific,
    actionable message; V1 never silently ignores a bad input."""


@dataclass(frozen=True)
class _Occupancy:
    """What one buffer occupies inside one tile: per buffer dimension, every
    extent some access needs there. The dimension is as wide as the widest
    of them, so two accesses to one buffer are counted once and the count is
    exact for a box-shaped access instead of a sum over accesses."""

    buffer: str
    dims: tuple[tuple[AxisExtent, ...], ...]
    elem_bytes: int


@dataclass(frozen=True)
class _Facts:
    """Everything the decisions read, all of it measured off ``tg``. Every
    per-statement field is indexed by statement name and, where it has a
    per-dimension shape, by that statement's own dimension. ``held`` and
    ``distances`` are then per statement per buffer; ``coincident[s]`` is the
    dimensions of ``s`` that carry no dependence."""

    extents: dict[str, tuple[int, ...]]
    coincident: dict[str, tuple[int, ...]]
    candidates: dict[str, tuple[AtomFact, ...]]
    held: dict[str, dict[str, _Occupancy]]
    distances: dict[str, dict[str, tuple[int, ...]]]
    capacity: int


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


def _candidates_for(
    unit: TileUnit, target: Target, stage: str
) -> tuple[AtomFact, ...]:
    """The atoms *target* admits for one statement, at *stage*.

    An operation the target's catalogue was never meant to cover is downgraded
    to "no candidates" rather than aborting the whole run: a statement with
    nothing to pick from is a schedule with one fewer decision in it, not a
    failure.
    """
    try:
        facts = TARGET_FACTS.project(
            target, AtomCandidateFacts, AtomCandidateQuery(stage=stage, op=unit.op)
        )
    except NotImplementedError:
        return ()
    # A target that does not schedule this level at all is a different failure
    # from an operation it has no atoms for, and it reports itself as one of
    # these rather than escaping as a bare projection error.
    except (TargetFactsError, TypeError, ValueError) as error:
        raise AtomSelectionError(
            f"select_atoms: target {target.name!r} states no atom candidates "
            f"at stage {stage!r}: {error}"
        ) from error
    return facts.candidates


def _tile_capacity(target: Target, stage: str) -> int:
    """The store a tile of the level being decided lives in.

    It is recorded against the footprint rather than enforced: an atom that
    cannot hold its operands is filtered out of the candidates, and a tile too
    wide for the store still has a schedule, only a worse one.
    """
    try:
        facts = TARGET_FACTS.project(target, TileStoreFacts, stage)
    except (TargetFactsError, TypeError, ValueError) as error:
        raise AtomSelectionError(
            f"select_atoms: target {target.name!r} states no tile-memory "
            f"capacity at stage {stage!r}: {error}"
        ) from error
    capacity = facts.tile_capacity_bytes
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise AtomSelectionError(
            f"select_atoms: stage {stage!r} states a tile-memory capacity of "
            f"{capacity!r}, which is not a positive byte count"
        )
    return capacity


def _atom_shape(name: str, rank: int, fact: AtomFact) -> tuple[int, ...]:
    """The atom's own extent per dimension of statement ``name``. An atom
    shape aligns to the *trailing* dimensions of the domain -- a 16x16x1 MNK
    atom granularises ``[m, n, k]`` and says nothing about the batch
    dimensions a batched matmul iterates outside them, which take extent 1."""
    if len(fact.shape) > rank:
        raise AtomSelectionError(
            f"select_atoms: statement {name!r} spans {rank} dimension(s) but "
            f"its candidate atom shape {fact.shape} has {len(fact.shape)} -- an "
            "atom cannot granularise a domain narrower than itself"
        )
    return (1,) * (rank - len(fact.shape)) + tuple(fact.shape)


def _held_by_statement(
    footprints: tuple[AccessFootprint, ...],
) -> dict[str, dict[str, _Occupancy]]:
    """Group ``footprints`` by statement then by buffer, widening each buffer
    dimension to every extent any access in that group needs there."""
    dims: dict[tuple[str, str], list[dict[AxisExtent, None]]] = {}
    bytes_of: dict[tuple[str, str], int] = {}
    for fp in footprints:
        group = (fp.statement, fp.buffer)
        if group not in dims:
            dims[group] = [{} for _ in fp.dims]
            bytes_of[group] = fp.elem_bytes
        if len(dims[group]) != len(fp.dims):
            raise AtomSelectionError(
                f"select_atoms: buffer {fp.buffer!r} is accessed with "
                f"{len(fp.dims)} and {len(dims[group])} dimension(s)"
            )
        for pos, extent in enumerate(fp.dims):
            dims[group][pos][extent] = None
    held: dict[str, dict[str, _Occupancy]] = {}
    for (stmt, buf), per_dim in dims.items():
        held.setdefault(stmt, {})[buf] = _Occupancy(
            buffer=buf,
            dims=tuple(tuple(options) for options in per_dim),
            elem_bytes=bytes_of[(stmt, buf)],
        )
    return held


def _band_of(tg: TileGraph) -> dict[str, "isl.schedule_node_band"]:
    """One band per statement, keyed by the statement it schedules."""
    found = schedule_bands(tg.tree)
    bands = {band_statement(band): band for band in found}
    missing = sorted(unit.name for unit in tg.units if unit.name not in bands)
    if missing:
        raise AtomSelectionError(
            f"select_atoms: statements {missing} have no band in tg.tree "
            f"(it carries {sorted(bands)}) -- both must come from one run"
        )
    if len(found) != len(tg.units):
        raise AtomSelectionError(
            f"select_atoms: tg.tree carries {len(found)} band(s) for "
            f"{len(tg.units)} statement(s) -- an already tiled tree has two, and "
            "resources are decided over the untiled one"
        )
    return bands


def _is_own_identity(dims: tuple[int, ...], extents: tuple[int, ...]) -> bool:
    """Whether a band schedules its statement's own dimensions, in order.
    A one-point dimension reads as constant (isl sees ``d = 0``, not ``d``),
    which is the identity there too."""
    return all(
        dim == pos or (dim == -1 and extents[pos] == 1) for pos, dim in enumerate(dims)
    )


def _collect_facts(tg: TileGraph, target: Target, stage: str) -> _Facts:
    bands = _band_of(tg)
    extents: dict[str, tuple[int, ...]] = {}
    coincident: dict[str, tuple[int, ...]] = {}
    distances: dict[str, dict[str, tuple[int, ...]]] = {}
    time_maps = isl.union_map("{}")
    for unit in tg.units:
        name = unit.name
        time_map = bands[name].get_partial_schedule_union_map()
        extents[name] = time_extents(tg, time_map)
        rank = len(extents[name])
        dims = statement_time_dims(tg, time_map)[name]
        if not _is_own_identity(dims, extents[name]):
            raise AtomSelectionError(
                f"select_atoms: the band of statement {name!r} schedules "
                f"dimensions {dims}, not its own in order -- select_atoms "
                "decides per domain dimension, which needs an identity band"
            )
        parallel = tg.parallel_dims.get(name, ())
        if len(parallel) != rank:
            raise AtomSelectionError(
                f"select_atoms: tg.parallel_dims carries {len(parallel)} flag(s) for "
                f"the {rank} dimension(s) of statement {name!r} -- extract fills one "
                "per dimension, and placement is read off them"
            )
        coincident[name] = tuple(d for d, is_parallel in enumerate(parallel) if is_parallel)
        distances[name] = {
            buf: carried
            for buf, carried in carried_distances(tg, time_map, rank).items()
            if any(carried)
        }
        time_maps = time_maps.union(time_map)

    return _Facts(
        extents=extents,
        coincident=coincident,
        candidates={
            unit.name: _candidates_for(unit, target, stage) for unit in tg.units
        },
        held=_held_by_statement(access_footprints(tg, time_maps)),
        distances=distances,
        capacity=_tile_capacity(target, stage),
    )


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _occupancy_bytes(occupancy: _Occupancy) -> int:
    """Bytes ``occupancy``'s buffer holds for one instance of its statement --
    the product of the measured per-dimension extents. Each dimension is a
    measured element count, so nothing here scales it by a tile size: the
    statement's own extent *is* its tile."""
    count = math.prod(
        max(extent.extent for extent in options) for options in occupancy.dims
    )
    return count * occupancy.elem_bytes


def _atom_instances(extents: tuple[int, ...], shape: tuple[int, ...]) -> int:
    """How many instances of an atom of ``shape`` cover a whole ``extents``."""
    return math.prod(_ceil_div(e, s) for e, s in zip(extents, shape))


def _statement_units(
    extents: tuple[int, ...], fact: AtomFact | None, shape: tuple[int, ...] | None
) -> int:
    """One statement's whole nominal time in duration units: its atom's own
    roofline estimate once per instance covering the domain, or the domain
    volume at the default per-element cost when it has no atom."""
    if fact is None or shape is None:
        return max(1, math.prod(extents) * _DEFAULT_UNITS)
    return max(1, round(fact.duration * _DURATION_SCALE)) * _atom_instances(extents, shape)


def _holders(facts: _Facts) -> dict[str, tuple[str, ...]]:
    """Per buffer, the statements holding it -- one buffer's ring is shared
    by every statement that touches it."""
    out: dict[str, list[str]] = {}
    for name, group in facts.held.items():
        for buf in group:
            out.setdefault(buf, []).append(name)
    return {buf: tuple(stmts) for buf, stmts in sorted(out.items())}


def _ring_depth(facts: _Facts, buf: str, holders: tuple[str, ...]) -> int:
    """One buffer's ring depth, measured rather than searched for: a
    dependence carried ``distance`` iterations along a dimension tiled
    ``tile`` wide spans ``ceil(distance / tile)`` tiles, and the ring holds
    one slot more than that so the older tile is still alive."""
    depths = [1]
    for name in holders:
        tile = facts.extents[name]
        for d, distance in enumerate(facts.distances[name].get(buf, ())):
            depths.append(_ceil_div(distance, tile[d]) + 1)
    return max(depths)


def _statement_precedence_pairs(tg: TileGraph) -> set[tuple[str, str]]:
    """``tg.deps`` coarsened to statement granularity: ``(a, b)`` iff some
    instance of ``a`` must execute before some instance of ``b``. A
    same-statement dependence constrains instance order inside one
    statement, so it is dropped."""
    maps: list["isl.map"] = []
    tg.deps.foreach_map(maps.append)
    pairs: set[tuple[str, str]] = set()
    for m in maps:
        a = m.get_tuple_name(isl.dim_type.IN)
        b = m.get_tuple_name(isl.dim_type.OUT)
        if a != b:
            pairs.add((a, b))
    return pairs


def _check_precedence(tg: TileGraph, order: list[str]) -> None:
    """``tg.units`` is the order ``tg.tree`` sequences the statements in and
    the nominal timeline is a prefix sum over it, so every dependence isl
    reports between two statements has to run with that order."""
    position = {name: pos for pos, name in enumerate(order)}
    for a, b in sorted(_statement_precedence_pairs(tg)):
        if a not in position or b not in position:
            raise AtomSelectionError(
                f"select_atoms: tg.deps references statement {a!r}->{b!r} "
                "not present in tg.units"
            )
        if position[a] >= position[b]:
            raise AtomSelectionError(
                f"select_atoms: tg.deps orders {a!r} before {b!r}, but tg.units "
                f"sequences them as {order} -- the timeline follows that order"
            )


def _place_of(facts: _Facts, name: str) -> str:
    """Placement is derived, never solved: this statement's own
    dependence-free dimensions are the ones spread over lanes."""
    members = facts.coincident[name]
    if not members:
        return "serial"
    return "coincident[" + ",".join(str(d) for d in members) + "]"


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def select_atoms(
    tg: TileGraph,
    target: Target | str | None = None,
    stage: str = "cta",
) -> TileGraph:
    """Decide, per statement of ``tg`` (which must already carry an isl
    schedule tree from ``build_schedule_tree``), which candidate atom
    granularises it, then tile every band by that statement's own extent.

    Raises :class:`AtomSelectionError` naming the cause when ``tg`` is not
    consistent with itself or the stage exposes no capacity fact.
    """
    if not tg.units:
        raise AtomSelectionError(
            "select_atoms: tg.units is empty -- nothing to decide resources for"
        )
    if tg.tree is None:
        raise AtomSelectionError(
            "select_atoms: tg.tree is None -- call build_schedule_tree(tg) first"
        )
    target = default_target() if target is None else resolve_target(target)

    facts = _collect_facts(tg, target, stage)
    order = [unit.name for unit in tg.units]
    _check_precedence(tg, order)

    tiles = {name: facts.extents[name] for name in order}
    ring = {buf: _ring_depth(facts, buf, holders) for buf, holders in _holders(facts).items()}

    statements: dict[str, dict] = {}
    clock = 0
    for name in order:
        extents = facts.extents[name]
        candidates = facts.candidates[name]
        # No cost model ranks two atoms yet, so the pick is the first candidate
        # that survived the catalogue's own filter; every survivor is recorded.
        picked = candidates[0] if candidates else None
        shape = _atom_shape(name, len(extents), picked) if picked else None
        duration = _statement_units(extents, picked, shape)
        footprint = {
            buf: _occupancy_bytes(occupancy) * ring[buf]
            for buf, occupancy in facts.held[name].items()
        }
        statements[name] = {
            "atom": picked.atom.op.name if picked else None,
            "candidates": tuple(fact.atom.op.name for fact in candidates),
            "place": _place_of(facts, name),
            "tile": extents,
            "tiles": (1,) * len(extents),
            "coincident": facts.coincident[name],
            "tile_atoms": _atom_instances(extents, shape) if shape else None,
            "duration": duration,
            "start": clock,
            "end": clock + duration,
            "footprint_bytes": footprint,
            "fits_capacity": sum(footprint.values()) <= facts.capacity,
        }
        clock += duration

    decisions = {
        # The decision space is a single point per statement, so the atom each
        # one takes is trivially the best there is.
        "status": "OPTIMAL",
        "makespan": clock,
        "capacity_bytes": facts.capacity,
        "statements": statements,
        "ring": ring,
    }
    return dataclasses.replace(
        tg,
        tree=tile_bands(tg.tree, tiles),
        ring=ring,
        decisions=decisions,
    )


__all__ = ["AtomSelectionError", "select_atoms"]
