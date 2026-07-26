"""``solve_resources(tg, target, options, stage) -> TileGraph`` -- the
CP-SAT resource decision over one scheduled ``TileGraph``.

Decided together: per statement one atom and one tile size per own
dimension that is a whole multiple of the picked atom's extent there, one
split of every statement's parallel extent across lanes, and per buffer a
ring depth. The objective is a deps-chain makespan whose per-tile stage
time is ``compute + load - hidden``, so a deeper ring buys load overlap
and a bigger tile amortises the load, both paid for against the capacity
fact. Parallel placement is not solved: it is read off
``tg.parallel_dims``. Returns ``tg`` with the tiled tree, ``ring`` and
``decisions``.

``build_schedule_tree`` gives each statement its own identity band, so a
band member *is* a domain dimension of the one statement under it -- every
fact and every variable below is per statement, per own dimension, with no
band-member-to-domain-dimension mapping in between.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import isl
from ortools.sat.python import cp_model

from tilefoundry.analysis import Analysis, AtomFact
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

from . import ScheduleOptions
from .kernel_schedule import band_statement, schedule_bands, tile_bands

# ns -> integer CP-SAT duration units. An atom's own roofline estimate is
# floored at 1.0ns; a bare round(ns) would flatten every candidate to the
# same "1" and make atom choice cost-indifferent inside CP-SAT.
_DURATION_SCALE = 1000

# Nominal per-domain-element cost (ns) for a statement with no registered
# atom candidate, whose only "how much work" signal is its own extent.
_DEFAULT_DURATION_NS = 1.0
_DEFAULT_UNITS = round(_DEFAULT_DURATION_NS * _DURATION_SCALE)

# `lane` count fallback for a target exposing no parallel-unit device fact.
_LANES_FALLBACK = 4

# Device fact names read, in order, for the two facts the model needs.
_CAPACITY_FACTS = ("shared_memory_per_cta_bytes", "l1_capacity_bytes")
_BANDWIDTH_FACTS = ("hbm_bandwidth_bytes_per_second", "l2_bandwidth_bytes_per_second")


class SolveResourcesError(RuntimeError):
    """CP-SAT reported an infeasible or otherwise unusable resource model,
    or a ``tg`` consistency precondition did not hold -- always raised with
    a specific, actionable message; V1 never silently ignores a bad solve."""


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
    """Everything the CP-SAT model reads, all of it measured before any
    variable exists. Every per-statement field is indexed by statement name
    and, where it has a per-dimension shape, by that statement's own
    dimension. ``held``/``distances`` are then per statement per buffer;
    ``coincident[s]`` is the dimensions of ``s`` that carry no dependence."""

    extents: dict[str, tuple[int, ...]]
    min_tile: dict[str, tuple[int, ...]]
    coincident: dict[str, tuple[int, ...]]
    candidates: dict[str, tuple[AtomFact, ...]]
    held: dict[str, dict[str, _Occupancy]]
    loaded: dict[str, tuple[_Occupancy, ...]]
    reads_of: dict[str, tuple[str, ...]]
    readers_of: dict[str, tuple[str, ...]]
    distances: dict[str, dict[str, tuple[int, ...]]]
    capacity: int
    bandwidth: int
    lanes: int


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


def _analysis_service(target: Target, stage: str) -> Analysis:
    try:
        return target.service(Analysis, stage)
    except (TypeError, ValueError) as error:
        raise SolveResourcesError(
            f"solve_resources: target {target.name!r} binds no Analysis "
            f"service at stage {stage!r}: {error}"
        ) from error


def _candidates_for(unit: TileUnit, analysis: Analysis) -> tuple[AtomFact, ...]:
    """``analysis.candidate_atoms``, made robust to the statements the
    target's catalogue was never meant to cover: an uncovered op is
    downgraded to "no candidates" so it never aborts the whole solve."""
    try:
        return tuple(analysis.candidate_atoms(unit.op))
    except NotImplementedError:
        return ()


def _device_fact(target: Target, names: tuple[str, ...], what: str) -> int:
    device = getattr(target, "device", None)
    for name in names:
        value = getattr(device, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise SolveResourcesError(
        f"solve_resources: target {target.name!r} exposes no {what} fact "
        f"(looked for {', '.join(names)} on its device)"
    )


def _lane_count(target: Target) -> int:
    device = getattr(target, "device", None)
    sm_count = getattr(device, "sm_count", None)
    if isinstance(sm_count, int) and not isinstance(sm_count, bool) and sm_count > 0:
        return sm_count
    return _LANES_FALLBACK


def _atom_shape(name: str, rank: int, fact: AtomFact) -> tuple[int, ...]:
    """The atom's own extent per dimension of statement ``name`` -- its own
    shape, which an identity band matches dimension for dimension."""
    if len(fact.shape) != rank:
        raise SolveResourcesError(
            f"solve_resources: statement {name!r} spans {rank} dimension(s) but "
            f"its candidate atom shape {fact.shape} has {len(fact.shape)} -- an "
            "atom can only granularise a domain of its own rank"
        )
    return tuple(fact.shape)


def _merge(footprints: tuple[AccessFootprint, ...], key) -> dict[object, dict[str, _Occupancy]]:
    """Group ``footprints`` by ``key`` then by buffer, widening each buffer
    dimension to every extent any access in that group needs there."""
    dims: dict[object, list[dict[AxisExtent, None]]] = {}
    bytes_of: dict[object, int] = {}
    for fp in footprints:
        group = (key(fp), fp.buffer)
        if group not in dims:
            dims[group] = [{} for _ in fp.dims]
            bytes_of[group] = fp.elem_bytes
        if len(dims[group]) != len(fp.dims):
            raise SolveResourcesError(
                f"solve_resources: buffer {fp.buffer!r} is accessed with "
                f"{len(fp.dims)} and {len(dims[group])} dimension(s)"
            )
        for pos, extent in enumerate(fp.dims):
            dims[group][pos][extent] = None
    merged: dict[object, dict[str, _Occupancy]] = {}
    for (group, buf), per_dim in dims.items():
        merged.setdefault(group, {})[buf] = _Occupancy(
            buffer=buf,
            dims=tuple(tuple(options) for options in per_dim),
            elem_bytes=bytes_of[(group, buf)],
        )
    return merged


def _min_tile(name: str, rank: int, candidates: tuple[AtomFact, ...]) -> tuple[int, ...]:
    """The smallest tile any solution can take for one statement: a whole
    multiple of its chosen atom's extent, so per dimension the smallest of
    its candidates' extents there (1 when it has no candidate)."""
    if not candidates:
        return (1,) * rank
    shapes = [_atom_shape(name, rank, c) for c in candidates]
    return tuple(min(shape[d] for shape in shapes) for d in range(rank))


def _band_of(tg: TileGraph) -> dict[str, "isl.schedule_node_band"]:
    """One band per statement, keyed by the statement it schedules."""
    found = schedule_bands(tg.tree)
    bands = {band_statement(band): band for band in found}
    missing = sorted(unit.name for unit in tg.units if unit.name not in bands)
    if missing:
        raise SolveResourcesError(
            f"solve_resources: statements {missing} have no band in tg.tree "
            f"(it carries {sorted(bands)}) -- both must come from one run"
        )
    if len(found) != len(tg.units):
        raise SolveResourcesError(
            f"solve_resources: tg.tree carries {len(found)} band(s) for "
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


def _collect_facts(tg: TileGraph, target: Target, analysis: Analysis) -> _Facts:
    bands = _band_of(tg)
    candidates = {unit.name: _candidates_for(unit, analysis) for unit in tg.units}
    extents: dict[str, tuple[int, ...]] = {}
    min_tile: dict[str, tuple[int, ...]] = {}
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
            raise SolveResourcesError(
                f"solve_resources: the band of statement {name!r} schedules "
                f"dimensions {dims}, not its own in order -- solve_resources "
                "decides per domain dimension, which needs an identity band"
            )
        min_tile[name] = _min_tile(name, rank, candidates[name])
        parallel = tg.parallel_dims.get(name, ())
        if len(parallel) != rank:
            raise SolveResourcesError(
                f"solve_resources: tg.parallel_dims carries {len(parallel)} flag(s) for "
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

    footprints = access_footprints(tg, time_maps)
    reads = tuple(fp for fp in footprints if fp.is_read)
    reads_of: dict[str, set[str]] = {}
    readers_of: dict[str, set[str]] = {}
    for fp in reads:
        reads_of.setdefault(fp.statement, set()).add(fp.buffer)
        readers_of.setdefault(fp.buffer, set()).add(fp.statement)

    return _Facts(
        extents=extents,
        min_tile=min_tile,
        coincident=coincident,
        candidates=candidates,
        held=_merge(footprints, lambda fp: fp.statement),
        loaded={
            stmt: tuple(group.values())
            for stmt, group in _merge(reads, lambda fp: fp.statement).items()
        },
        reads_of={s: tuple(sorted(bufs)) for s, bufs in reads_of.items()},
        readers_of={b: tuple(sorted(stmts)) for b, stmts in readers_of.items()},
        distances=distances,
        capacity=_device_fact(target, _CAPACITY_FACTS, "tile-memory capacity"),
        bandwidth=_device_fact(target, _BANDWIDTH_FACTS, "memory bandwidth"),
        lanes=_lane_count(target),
    )


def _extent_value(extent: AxisExtent, tile: tuple[int, ...]) -> int:
    return extent.constant * math.prod(tile[axis] for axis in extent.axes)


def _occupancy_bytes(occupancy: _Occupancy, tile: tuple[int, ...]) -> int:
    count = math.prod(
        max(_extent_value(extent, tile) for extent in options) for options in occupancy.dims
    )
    return count * occupancy.elem_bytes


def _total_bytes(group, tile: tuple[int, ...]) -> int:
    return sum(_occupancy_bytes(occupancy, tile) for occupancy in group)


def _compute_units(fact: AtomFact) -> int:
    """The atom's compute-side cost only. ``AtomFact.duration`` is a
    roofline max over compute and the atom's own traffic; the tile's traffic
    is modelled separately here, so charging the max would count memory
    twice."""
    return max(1, round(fact.compute_duration * _DURATION_SCALE))


def _bandwidth_ratio(bandwidth: int) -> tuple[int, int]:
    """``(num, den)`` with ``units == bytes * den / num``: a duration unit is
    ``1/_DURATION_SCALE`` ns, so ``units = bytes * 1e9 * scale / bandwidth``."""
    num, den = bandwidth, 10**9 * _DURATION_SCALE
    common = math.gcd(num, den)
    return num // common, den // common


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# CP-SAT model
# ---------------------------------------------------------------------------


def _product(model: cp_model.CpModel, factors: list, ub: int, name: str):
    if not factors:
        return 1
    if len(factors) == 1:
        return factors[0]
    var = model.NewIntVar(1, ub, name)
    model.AddMultiplicationEquality(var, factors)
    return var


def _bytes_expr(model: cp_model.CpModel, group, tile: list, extents: tuple[int, ...], name: str):
    """The bytes ``group``'s occupancies hold in one tile of the statement
    whose dimensions ``tile``/``extents`` describe: per buffer dimension the
    widest extent any access needs, multiplied out."""
    parts = []
    for occupancy in group:
        factors = []
        for pos, options in enumerate(occupancy.dims):
            ub = max(_extent_value(extent, extents) for extent in options)
            widths = [
                extent.constant * _product(
                    model, [tile[d] for d in extent.axes],
                    _extent_value(extent, extents),
                    f"tile_{name}_{occupancy.buffer}_{pos}_{i}",
                )
                for i, extent in enumerate(options)
            ]
            if len(widths) == 1:
                factors.append(widths[0])
                continue
            widest = model.NewIntVar(1, ub, f"wide_{name}_{occupancy.buffer}_{pos}")
            model.AddMaxEquality(widest, widths)
            factors.append(widest)
        count = _product(
            model, factors, _occupancy_bytes(occupancy, extents) // occupancy.elem_bytes,
            f"elems_{name}_{occupancy.buffer}",
        )
        parts.append(occupancy.elem_bytes * count)
    return sum(parts) if parts else 0


@dataclass(frozen=True)
class _Plan:
    """One statement's variables. ``candidates``/``pick``/``mult`` are empty
    together: no atom then constrains the tile for this statement."""

    name: str
    candidates: tuple[AtomFact, ...]
    pick: tuple[cp_model.IntVar, ...]
    mult: tuple[cp_model.IntVar, ...]
    compute: cp_model.IntVar
    load: cp_model.IntVar
    stage: cp_model.IntVar
    chunks: cp_model.IntVar
    start: cp_model.IntVar
    end: cp_model.IntVar


def _compute_bound(facts: _Facts, name: str) -> int:
    """One statement's whole compute cost, which is tile-independent: the
    atom count over its domain times the atom's own duration, or the domain
    volume at the default per-element cost."""
    extents = facts.extents[name]
    cands = facts.candidates[name]
    if not cands:
        return math.prod(extents) * _DEFAULT_UNITS
    rank = len(extents)
    return max(
        _compute_units(c)
        * math.prod(_ceil_div(extents[d], shape[d]) for d in range(rank))
        for c, shape in ((c, _atom_shape(name, rank, c)) for c in cands)
    )


def _reload_factor(occupancy: _Occupancy, extents: tuple[int, ...]) -> int:
    """How many times over one buffer's whole traffic a growing tile can
    re-read it: a buffer dimension no tile dimension reaches shrinks the
    traffic (fewer tiles, same bytes), one a single tile dimension reaches
    leaves it flat, and a tile dimension two buffer dimensions share grows
    it by that dimension's extent."""
    reached = [0] * len(extents)
    for options in occupancy.dims:
        for axis in {a for extent in options for a in extent.axes}:
            reached[axis] += 1
    return math.prod(extents[d] ** (n - 1) for d, n in enumerate(reached) if n > 1)


@dataclass(frozen=True)
class _Bounds:
    """The largest value each duration variable can take. ``compute``/``load``
    are per tile, ``span`` is one statement's whole time and ``horizon`` their
    sum -- the statements are sequenced, so that bounds the makespan. Every
    variable gets the tightest of these it can, so no CP-SAT product has to
    range over the whole horizon."""

    compute: dict[str, int]
    load: dict[str, int]
    span: dict[str, int]
    horizon: int

    def stage(self, name: str) -> int:
        return self.compute[name] + self.load[name]


def _bounds(facts: _Facts) -> _Bounds:
    num, den = _bandwidth_ratio(facts.bandwidth)
    compute: dict[str, int] = {}
    load: dict[str, int] = {}
    span: dict[str, int] = {}
    for name, extents in facts.extents.items():
        loaded = facts.loaded.get(name, ())
        instances = math.prod(extents) or 1
        ones = (1,) * len(extents)
        # The whole domain in one tile: the most bytes one tile can read.
        whole = _ceil_div(_total_bytes(loaded, extents) * den, num) + 1
        # Every tile at its finest: the most bytes every tile can read summed.
        total = sum(
            _occupancy_bytes(occupancy, ones) * instances * _reload_factor(occupancy, extents)
            for occupancy in loaded
        )
        compute[name] = _compute_bound(facts, name)
        load[name] = whole
        span[name] = compute[name] + _ceil_div(total * den, num) + instances
    return _Bounds(
        compute=compute, load=load, span=span, horizon=max(1, sum(span.values()))
    )


def _plan_statements(
    model: cp_model.CpModel, facts: _Facts, tiles: dict[str, list], bounds: _Bounds
) -> dict[str, _Plan]:
    num, den = _bandwidth_ratio(facts.bandwidth)
    horizon = bounds.horizon
    plans: dict[str, _Plan] = {}
    for name, extents in facts.extents.items():
        rank = len(extents)
        tile = tiles[name]
        cands = facts.candidates[name]
        volume_ub = math.prod(extents) or 1
        mult: list[cp_model.IntVar] = []
        compute = model.NewIntVar(1, bounds.compute[name], f"compute_{name}")
        if cands:
            pick = tuple(model.NewBoolVar(f"pick_{name}_{i}") for i in range(len(cands)))
            model.AddExactlyOne(pick)
            shapes = [_atom_shape(name, rank, c) for c in cands]
            for d in range(rank):
                sizes = [shape[d] for shape in shapes]
                atom = model.NewIntVar(min(sizes), max(sizes), f"atom_{name}_{d}")
                model.Add(atom == sum(p * s for p, s in zip(pick, sizes)))
                factor = model.NewIntVar(1, extents[d], f"mult_{name}_{d}")
                model.AddMultiplicationEquality(tile[d], [atom, factor])
                mult.append(factor)
            units = [_compute_units(c) for c in cands]
            per_tile = model.NewIntVar(min(units), max(units), f"atom_units_{name}")
            model.Add(per_tile == sum(p * u for p, u in zip(pick, units)))
            atoms = _product(model, mult, volume_ub, f"atoms_{name}")
            model.AddMultiplicationEquality(compute, [per_tile, atoms])
        else:
            pick = ()
            volume = _product(model, list(tile), volume_ub, f"volume_{name}")
            model.Add(compute == volume * _DEFAULT_UNITS)

        read_bytes = _bytes_expr(
            model, facts.loaded.get(name, ()), tile, extents, f"read_{name}"
        )
        load = model.NewIntVar(0, bounds.load[name], f"load_{name}")
        model.AddDivisionEquality(load, read_bytes * den + num - 1, num)
        plans[name] = _Plan(
            name=name, candidates=cands, pick=pick, mult=tuple(mult), compute=compute, load=load,
            stage=model.NewIntVar(1, bounds.stage(name), f"stage_{name}"),
            chunks=model.NewIntVar(1, volume_ub, f"chunks_{name}"),
            start=model.NewIntVar(0, horizon, f"start_{name}"),
            end=model.NewIntVar(0, horizon, f"end_{name}"),
        )
    return plans


def _holders(facts: _Facts) -> dict[str, tuple[str, ...]]:
    """Per buffer, the statements holding it -- one buffer's ring is shared
    by every statement that touches it, each at its own tile."""
    out: dict[str, list[str]] = {}
    for name, group in facts.held.items():
        for buf in group:
            out.setdefault(buf, []).append(name)
    return {buf: tuple(stmts) for buf, stmts in sorted(out.items())}


def _plan_rings(
    model: cp_model.CpModel, facts: _Facts, plans: dict[str, _Plan], tiles: dict[str, list]
) -> tuple[dict[str, cp_model.IntVar], dict[str, int]]:
    """Per buffer a ring depth: at least one slot more than the tile
    distance isl reports for the dependence it carries, deeper only when it
    carries such a dependence or an async atom produces it, and in every
    case paid for in the capacity sum. The upper bound is what the capacity
    fact can hold at the smallest tile of its widest holder."""
    async_of = {
        name: (
            sum(p for p, c in zip(plan.pick, plan.candidates) if c.is_async)
            if plan.pick else 0
        )
        for name, plan in plans.items()
    }
    ring: dict[str, cp_model.IntVar] = {}
    uppers: dict[str, int] = {}
    for buf, holders in _holders(facts).items():
        smallest = max(
            1,
            max(_occupancy_bytes(facts.held[s][buf], facts.min_tile[s]) for s in holders),
        )
        carried = max(
            (max(facts.distances[s].get(buf, (0,))) for s in holders), default=0
        )
        upper = max(1, facts.capacity // smallest, carried + 1)
        depth = model.NewIntVar(1, upper, f"ring_{buf}")
        for s in holders:
            for d, distance in enumerate(facts.distances[s].get(buf, ())):
                if distance == 0:
                    continue
                steps = model.NewIntVar(1, distance, f"carry_{buf}_{s}_{d}")
                model.AddDivisionEquality(steps, distance + tiles[s][d] - 1, tiles[s][d])
                model.Add(depth >= steps + 1)
        usable = model.NewBoolVar(f"ring_ok_{buf}")
        model.AddMaxEquality(
            usable,
            [int(carried > 0)] + [async_of[s] for s in facts.readers_of.get(buf, ())],
        )
        model.Add(depth == 1).OnlyEnforceIf(usable.Not())
        ring[buf] = depth
        uppers[buf] = upper
    return ring, uppers


def _pipeline(
    model: cp_model.CpModel, facts: _Facts, plans: dict[str, _Plan],
    ring: dict[str, cp_model.IntVar], uppers: dict[str, int], bounds: _Bounds,
) -> None:
    """``stage == compute + load - hidden``: the ring slots of every buffer a
    statement reads can hide at most that many extra tiles' worth of
    compute, so one slot hides nothing and each further slot hides one more
    tile of load latency."""
    for name, plan in plans.items():
        bufs = facts.reads_of.get(name, ())
        slots = max((uppers[b] for b in bufs), default=1)
        extra = model.NewIntVar(0, slots, f"extra_slots_{name}")
        if bufs:
            depth = model.NewIntVar(1, slots, f"depth_{name}")
            model.AddMinEquality(depth, [ring[buf] for buf in bufs])
            model.Add(extra == depth - 1)
        else:
            model.Add(extra == 0)
        hidden = model.NewIntVar(0, bounds.load[name], f"hidden_{name}")
        cap = model.NewIntVar(0, slots * bounds.compute[name], f"hide_cap_{name}")
        model.AddMultiplicationEquality(cap, [extra, plan.compute])
        model.AddMinEquality(hidden, [cap, plan.load])
        model.Add(plan.stage == plan.compute + plan.load - hidden)


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


def _capacity_precheck(facts: _Facts) -> None:
    """The smallest footprint any solution can have for each statement --
    every dimension at its smallest atom extent, every ring one slot deep.
    Over capacity here is over capacity for every tile of that statement, so
    it is reported as such rather than left to come back as a bare CP-SAT
    ``INFEASIBLE``."""
    for name, group in facts.held.items():
        per_buffer = {
            buf: _occupancy_bytes(occupancy, facts.min_tile[name])
            for buf, occupancy in group.items()
        }
        total = sum(per_buffer.values())
        if total > facts.capacity:
            detail = ", ".join(f"{buf}={n}" for buf, n in sorted(per_buffer.items()))
            raise SolveResourcesError(
                f"solve_resources: capacity {facts.capacity} bytes cannot hold one "
                f"atom tile {facts.min_tile[name]} of statement {name!r}, which "
                f"requests {total} bytes ({detail})"
            )


def _infeasible_message(facts: _Facts, status: str) -> str:
    smallest = {
        name: _total_bytes(group.values(), facts.min_tile[name])
        for name, group in facts.held.items()
    }
    carried = {name: dims for name, dims in facts.distances.items() if dims}
    uncovered = sorted(name for name, cands in facts.candidates.items() if not cands)
    return (
        f"solve_resources: CP-SAT returned {status!r}; capacity {facts.capacity} "
        f"bytes, smallest per-statement tile footprints {smallest}, carried "
        f"dependence distances {carried}, statements with no candidate atom "
        f"{uncovered}"
    )


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def solve_resources(
    tg: TileGraph,
    target: Target | str | None = None,
    options: ScheduleOptions | None = None,
    stage: str = "cta",
) -> TileGraph:
    """Decide, per statement, an atom and a tile size per own dimension, the
    lane split of every statement's parallel extent, and per buffer a ring
    depth over ``tg`` (which must already carry an isl schedule tree from
    ``build_schedule_tree``), then tile every band by its decided sizes.

    Raises :class:`SolveResourcesError` naming the cause when the model
    cannot be satisfied -- V1 reports, it never retries with another
    schedule.
    """
    if not tg.units:
        raise SolveResourcesError(
            "solve_resources: tg.units is empty -- nothing to solve resources for"
        )
    if tg.tree is None:
        raise SolveResourcesError(
            "solve_resources: tg.tree is None -- call build_schedule_tree(tg) first"
        )
    target = default_target() if target is None else resolve_target(target)
    options = options if options is not None else ScheduleOptions()

    facts = _collect_facts(tg, target, _analysis_service(target, stage))
    _capacity_precheck(facts)
    bounds = _bounds(facts)
    horizon = bounds.horizon

    model = cp_model.CpModel()
    tiles: dict[str, list] = {}
    counts: dict[str, list] = {}
    for name, extents in facts.extents.items():
        tiles[name] = [model.NewIntVar(1, e, f"tile_{name}_{d}") for d, e in enumerate(extents)]
        counts[name] = [model.NewIntVar(1, e, f"tiles_{name}_{d}") for d, e in enumerate(extents)]
        for d, extent in enumerate(extents):
            model.AddMultiplicationEquality(extent, [tiles[name][d], counts[name][d]])

    plans = _plan_statements(model, facts, tiles, bounds)
    ring, uppers = _plan_rings(model, facts, plans, tiles)
    for name, group in facts.held.items():
        footprint = []
        for buf, occupancy in group.items():
            held = model.NewIntVar(0, facts.capacity, f"held_{name}_{buf}")
            model.AddMultiplicationEquality(
                held,
                [
                    _bytes_expr(
                        model, (occupancy,), tiles[name], facts.extents[name], f"buf_{name}"
                    ),
                    ring[buf],
                ],
            )
            footprint.append(held)
        model.Add(sum(footprint) <= facts.capacity)
    _pipeline(model, facts, plans, ring, uppers, bounds)

    # The hardware bounds the lane split above; the round count each statement
    # divides out of it bounds the makespan. The objective's tie-break then
    # settles it at the fewest lanes reaching the best makespan, which is the
    # coincident tile count whenever the hardware has that many lanes.
    lane_split = model.NewIntVar(1, facts.lanes, "lane_split")
    for name, plan in plans.items():
        count = counts[name]
        coincident = facts.coincident[name]
        volume_ub = math.prod(facts.extents[name]) or 1
        parallel = _product(
            model, [count[d] for d in range(len(count)) if d in coincident],
            volume_ub, f"parallel_{name}",
        )
        serial = _product(
            model, [count[d] for d in range(len(count)) if d not in coincident],
            volume_ub, f"serial_{name}",
        )
        model.AddDivisionEquality(plan.chunks, parallel + lane_split - 1, lane_split)
        rounds = _product(model, [plan.chunks, serial], volume_ub, f"rounds_{name}")
        span = model.NewIntVar(1, bounds.span[name], f"span_{name}")
        model.AddMultiplicationEquality(span, [rounds, plan.stage])
        model.Add(plan.end == plan.start + span)

    for a, b in _statement_precedence_pairs(tg):
        if a not in plans or b not in plans:
            raise SolveResourcesError(
                f"solve_resources: tg.deps references statement {a!r}->{b!r} "
                "not present in tg.units"
            )
        model.Add(plans[b].start >= plans[a].end)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [plan.end for plan in plans.values()])
    # Lexicographic: the makespan first, then the fewest ring slots and lanes
    # that reach it, so neither a slot nothing overlaps nor a lane with no
    # tile on it is ever bought.
    slack = 1 + sum(uppers.values()) + facts.lanes
    model.Minimize(makespan * slack + sum(ring.values()) + lane_split)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = options.timeout_seconds
    solver.parameters.num_search_workers = options.workers
    solver.parameters.random_seed = options.random_seed
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise SolveResourcesError(_infeasible_message(facts, solver.StatusName(status)))

    sizes = {name: tuple(solver.Value(v) for v in tile) for name, tile in tiles.items()}
    ring_decisions = {buf: solver.Value(var) for buf, var in ring.items()}
    decisions = {
        "status": solver.StatusName(status),
        "makespan": solver.Value(makespan),
        "lanes": facts.lanes,
        "lane_split": solver.Value(lane_split),
        "capacity_bytes": facts.capacity,
        "statements": {
            name: {
                "atom": _picked_atom(solver, plan),
                "place": _place_of(facts, name),
                "tile": sizes[name],
                "tiles": tuple(solver.Value(v) for v in counts[name]),
                "coincident": facts.coincident[name],
                "tile_atoms": (
                    math.prod(solver.Value(v) for v in plan.mult) if plan.mult else None
                ),
                "compute": solver.Value(plan.compute),
                "load": solver.Value(plan.load),
                "stage": solver.Value(plan.stage),
                "lane_rounds": solver.Value(plan.chunks),
                "start": solver.Value(plan.start),
                "end": solver.Value(plan.end),
                "footprint_bytes": {
                    buf: _occupancy_bytes(occupancy, sizes[name]) * ring_decisions[buf]
                    for buf, occupancy in facts.held[name].items()
                },
            }
            for name, plan in plans.items()
        },
        "ring": ring_decisions,
    }
    return dataclasses.replace(
        tg,
        tree=tile_bands(tg.tree, sizes),
        ring=ring_decisions,
        decisions=decisions,
    )


def _picked_atom(solver: cp_model.CpSolver, plan: _Plan) -> str | None:
    if not plan.candidates:
        return None
    chosen = next(i for i in range(len(plan.candidates)) if solver.Value(plan.pick[i]))
    return plan.candidates[chosen].atom.op.name


def _place_of(facts: _Facts, name: str) -> str:
    """Placement is derived, never solved: this statement's own
    dependence-free dimensions are the ones spread over lanes."""
    members = facts.coincident[name]
    if not members:
        return "serial"
    return "coincident[" + ",".join(str(d) for d in members) + "]"


__all__ = ["SolveResourcesError", "solve_resources"]
