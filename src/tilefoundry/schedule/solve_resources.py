"""``solve_resources(tg, target, options, stage) -> TileGraph`` -- the
CP-SAT resource decision over one scheduled ``TileGraph``.

Decided together: per statement one atom, per band member a tile size that
is a whole multiple of the picked atom's own extent, one split of the
band's coincident extent across lanes, and per buffer a ring depth. The
objective is a deps-chain makespan whose per-tile stage time is
``compute + load - hidden``, so a deeper ring buys load overlap and a
bigger tile amortises the load, both paid for against the capacity fact.
Parallel placement is not solved: it is read off the band's ``coincident``
members. Returns ``tg`` with the tiled tree, ``ring`` and ``decisions``.
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
from .kernel_schedule import outermost_band, tile_band

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
    variable exists. ``dim_of[s][b]`` is the domain dimension of statement
    ``s`` that band member ``b`` travels with (``-1`` when constant);
    ``axes_of[s]`` is the members it varies in."""

    n_dims: int
    extents: tuple[int, ...]
    min_tile: tuple[int, ...]
    coincident: tuple[int, ...]
    axes_of: dict[str, tuple[int, ...]]
    dim_of: dict[str, tuple[int, ...]]
    candidates: dict[str, tuple[AtomFact, ...]]
    held: dict[str, _Occupancy]
    loaded: dict[str, tuple[_Occupancy, ...]]
    reads_of: dict[str, tuple[str, ...]]
    readers_of: dict[str, tuple[str, ...]]
    distances: dict[str, tuple[int, ...]]
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


def _atom_extents(dim_of: tuple[int, ...], axes: tuple[int, ...], name: str, fact: AtomFact):
    """The atom's own extent per band member: its shape indexed by the
    domain dimension that member travels with."""
    rank = max(dim_of) + 1
    if len(fact.shape) != rank:
        raise SolveResourcesError(
            f"solve_resources: statement {name!r} spans {rank} dimension(s) but "
            f"its candidate atom shape {fact.shape} has {len(fact.shape)} -- an "
            "atom can only granularise a domain of its own rank"
        )
    return {b: fact.shape[dim_of[b]] for b in axes}


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


def _collect_facts(tg: TileGraph, target: Target, analysis: Analysis) -> _Facts:
    band = outermost_band(tg.tree)
    time_map = band.get_partial_schedule_union_map()
    n_dims = band.n_member()
    extents = time_extents(tg, time_map)
    dim_of = statement_time_dims(tg, time_map)
    names = {unit.name for unit in tg.units}
    if set(dim_of) != names:
        raise SolveResourcesError(
            f"solve_resources: band covers statements {sorted(dim_of)} but "
            f"tg.units names {sorted(names)} -- both must come from one run"
        )
    axes_of = {s: tuple(b for b in range(n_dims) if row[b] >= 0) for s, row in dim_of.items()}
    candidates = {u.name: _candidates_for(u, analysis) for u in tg.units}

    # Any feasible tile is a whole multiple of every statement's chosen atom
    # extent, so the smallest one is per member the largest of the per
    # statement smallest candidate extents.
    min_tile = [1] * n_dims
    for name, cands in candidates.items():
        per_atom = [_atom_extents(dim_of[name], axes_of[name], name, c) for c in cands]
        if not per_atom:
            continue
        for b in axes_of[name]:
            min_tile[b] = max(min_tile[b], min(extents_of[b] for extents_of in per_atom))

    footprints = access_footprints(tg, time_map)
    reads = tuple(fp for fp in footprints if fp.is_read)
    reads_of: dict[str, set[str]] = {}
    readers_of: dict[str, set[str]] = {}
    for fp in reads:
        reads_of.setdefault(fp.statement, set()).add(fp.buffer)
        readers_of.setdefault(fp.buffer, set()).add(fp.statement)

    return _Facts(
        n_dims=n_dims,
        extents=extents,
        min_tile=tuple(min_tile),
        coincident=tuple(b for b in range(n_dims) if band.member_get_coincident(b)),
        axes_of=axes_of,
        dim_of=dim_of,
        candidates=candidates,
        held=_merge(footprints, lambda fp: None)[None],
        loaded={
            stmt: tuple(group.values())
            for stmt, group in _merge(reads, lambda fp: fp.statement).items()
        },
        reads_of={s: tuple(sorted(bufs)) for s, bufs in reads_of.items()},
        readers_of={b: tuple(sorted(stmts)) for b, stmts in readers_of.items()},
        distances=carried_distances(tg, time_map, n_dims),
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


def _bytes_expr(model: cp_model.CpModel, facts: _Facts, group, tile: list, name: str):
    """The bytes ``group``'s occupancies hold in one tile: per buffer
    dimension the widest extent any access needs, multiplied out."""
    parts = []
    for occupancy in group:
        factors = []
        for pos, options in enumerate(occupancy.dims):
            ub = max(_extent_value(extent, facts.extents) for extent in options)
            widths = [
                extent.constant * _product(
                    model, [tile[b] for b in extent.axes],
                    _extent_value(extent, facts.extents),
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
            model, factors, _occupancy_bytes(occupancy, facts.extents) // occupancy.elem_bytes,
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
    mult: dict[int, cp_model.IntVar]
    compute: cp_model.IntVar
    load: cp_model.IntVar
    stage: cp_model.IntVar
    chunks: cp_model.IntVar
    start: cp_model.IntVar
    end: cp_model.IntVar


def _horizon(facts: _Facts) -> int:
    """A static bound for every ``start``/``end``/``makespan``. A
    statement's total compute is tile-independent (tile count times atoms
    per tile is fixed by the atom); its total load is bounded by the most
    tiles it can have times its whole-domain traffic."""
    num, den = _bandwidth_ratio(facts.bandwidth)
    total = 0
    for name, axes in facts.axes_of.items():
        cands = facts.candidates[name]
        if cands:
            total += max(
                _compute_units(c) * math.prod(
                    _ceil_div(facts.extents[b], extent) for b, extent
                    in _atom_extents(facts.dim_of[name], axes, name, c).items()
                )
                for c in cands
            )
        else:
            total += math.prod(facts.extents[b] for b in axes) * _DEFAULT_UNITS
        tiles = math.prod(_ceil_div(facts.extents[b], facts.min_tile[b]) for b in axes)
        whole = _total_bytes(facts.loaded.get(name, ()), facts.extents)
        total += tiles * (_ceil_div(whole * den, num) + 1)
    return max(1, total)


def _plan_statements(
    model: cp_model.CpModel, facts: _Facts, tile: list, horizon: int
) -> dict[str, _Plan]:
    num, den = _bandwidth_ratio(facts.bandwidth)
    plans: dict[str, _Plan] = {}
    for name, axes in facts.axes_of.items():
        cands = facts.candidates[name]
        volume_ub = math.prod(facts.extents[b] for b in axes) or 1
        mult: dict[int, cp_model.IntVar] = {}
        compute = model.NewIntVar(1, horizon, f"compute_{name}")
        if cands:
            pick = tuple(model.NewBoolVar(f"pick_{name}_{i}") for i in range(len(cands)))
            model.AddExactlyOne(pick)
            per_atom = [_atom_extents(facts.dim_of[name], axes, name, c) for c in cands]
            for b in axes:
                sizes = [extents[b] for extents in per_atom]
                atom = model.NewIntVar(min(sizes), max(sizes), f"atom_{name}_{b}")
                model.Add(atom == sum(p * s for p, s in zip(pick, sizes)))
                mult[b] = model.NewIntVar(1, facts.extents[b], f"mult_{name}_{b}")
                model.AddMultiplicationEquality(tile[b], [atom, mult[b]])
            units = [_compute_units(c) for c in cands]
            per_tile = model.NewIntVar(min(units), max(units), f"atom_units_{name}")
            model.Add(per_tile == sum(p * u for p, u in zip(pick, units)))
            atoms = _product(model, [mult[b] for b in axes], volume_ub, f"atoms_{name}")
            model.AddMultiplicationEquality(compute, [per_tile, atoms])
        else:
            pick = ()
            volume = _product(model, [tile[b] for b in axes], volume_ub, f"volume_{name}")
            model.Add(compute == volume * _DEFAULT_UNITS)

        read_bytes = _bytes_expr(
            model, facts, facts.loaded.get(name, ()), tile, f"read_{name}"
        )
        load = model.NewIntVar(0, horizon, f"load_{name}")
        model.AddDivisionEquality(load, read_bytes * den + num - 1, num)
        plans[name] = _Plan(
            name=name, candidates=cands, pick=pick, mult=mult, compute=compute, load=load,
            stage=model.NewIntVar(1, horizon, f"stage_{name}"),
            chunks=model.NewIntVar(1, volume_ub, f"chunks_{name}"),
            start=model.NewIntVar(0, horizon, f"start_{name}"),
            end=model.NewIntVar(0, horizon, f"end_{name}"),
        )
    return plans


def _plan_rings(
    model: cp_model.CpModel, facts: _Facts, plans: dict[str, _Plan], tile: list
) -> tuple[dict[str, cp_model.IntVar], dict[str, int]]:
    """Per buffer a ring depth: at least one slot more than the tile
    distance isl reports for the dependence it carries, deeper only when it
    carries such a dependence or an async atom produces it, and in every
    case paid for in the capacity sum. The upper bound is what the capacity
    fact can hold at the smallest tile."""
    async_of = {
        name: (
            sum(p for p, c in zip(plan.pick, plan.candidates) if c.is_async)
            if plan.pick else 0
        )
        for name, plan in plans.items()
    }
    ring: dict[str, cp_model.IntVar] = {}
    uppers: dict[str, int] = {}
    for buf, occupancy in facts.held.items():
        distances = facts.distances.get(buf, (0,) * facts.n_dims)
        carried = max(distances, default=0)
        smallest = max(1, _occupancy_bytes(occupancy, facts.min_tile))
        upper = max(1, facts.capacity // smallest, carried + 1)
        depth = model.NewIntVar(1, upper, f"ring_{buf}")
        for b, distance in enumerate(distances):
            if distance == 0:
                continue
            steps = model.NewIntVar(1, distance, f"carry_{buf}_{b}")
            model.AddDivisionEquality(steps, distance + tile[b] - 1, tile[b])
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
    ring: dict[str, cp_model.IntVar], uppers: dict[str, int], horizon: int,
) -> None:
    """``stage == compute + load - hidden``: the ring slots of every buffer a
    statement reads can hide at most that many extra tiles' worth of
    compute, so one slot hides nothing and each further slot hides one more
    tile of load latency."""
    for name, plan in plans.items():
        bufs = facts.reads_of.get(name, ())
        extra = model.NewIntVar(0, max(uppers.values(), default=1), f"extra_slots_{name}")
        if bufs:
            depth = model.NewIntVar(1, max(uppers[b] for b in bufs), f"depth_{name}")
            model.AddMinEquality(depth, [ring[buf] for buf in bufs])
            model.Add(extra == depth - 1)
        else:
            model.Add(extra == 0)
        hidden = model.NewIntVar(0, horizon, f"hidden_{name}")
        cap = model.NewIntVar(0, horizon, f"hide_cap_{name}")
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
    """The smallest footprint any solution can have -- every band member at
    its smallest atom extent, every ring one slot deep. Over capacity here
    is over capacity for every tile, so it is reported as such rather than
    left to come back as a bare CP-SAT ``INFEASIBLE``."""
    per_buffer = {
        buf: _occupancy_bytes(occupancy, facts.min_tile)
        for buf, occupancy in facts.held.items()
    }
    total = sum(per_buffer.values())
    if total > facts.capacity:
        detail = ", ".join(f"{buf}={n}" for buf, n in sorted(per_buffer.items()))
        raise SolveResourcesError(
            f"solve_resources: capacity {facts.capacity} bytes cannot hold one "
            f"atom tile {facts.min_tile}, which requests {total} bytes ({detail})"
        )


def _infeasible_message(facts: _Facts, status: str) -> str:
    smallest = _total_bytes(facts.held.values(), facts.min_tile)
    carried = {buf: d for buf, d in sorted(facts.distances.items()) if any(d)}
    uncovered = sorted(name for name, cands in facts.candidates.items() if not cands)
    return (
        f"solve_resources: CP-SAT returned {status!r}; smallest tile "
        f"{facts.min_tile} needs {smallest} of {facts.capacity} capacity bytes, "
        f"carried dependence distances {carried}, statements with no candidate "
        f"atom {uncovered}"
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
    """Decide atoms, tile sizes, the lane split of the band's coincident
    extent and per-buffer ring depths over ``tg`` (which must already carry
    an isl schedule tree from ``compute_schedule``), then tile the band by
    the decided sizes.

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
            "solve_resources: tg.tree is None -- call compute_schedule(tg) first"
        )
    target = default_target() if target is None else resolve_target(target)
    options = options if options is not None else ScheduleOptions()

    facts = _collect_facts(tg, target, _analysis_service(target, stage))
    _capacity_precheck(facts)
    horizon = _horizon(facts)

    model = cp_model.CpModel()
    tile = [model.NewIntVar(1, facts.extents[b], f"tile_{b}") for b in range(facts.n_dims)]
    counts = [model.NewIntVar(1, facts.extents[b], f"tiles_{b}") for b in range(facts.n_dims)]
    for b in range(facts.n_dims):
        model.AddMultiplicationEquality(facts.extents[b], [tile[b], counts[b]])

    plans = _plan_statements(model, facts, tile, horizon)
    ring, uppers = _plan_rings(model, facts, plans, tile)
    footprint = []
    for buf, occupancy in facts.held.items():
        held = model.NewIntVar(0, facts.capacity, f"held_{buf}")
        model.AddMultiplicationEquality(
            held, [_bytes_expr(model, facts, (occupancy,), tile, "buf"), ring[buf]]
        )
        footprint.append(held)
    model.Add(sum(footprint) <= facts.capacity)
    _pipeline(model, facts, plans, ring, uppers, horizon)

    # The hardware bounds the lane split above; the round count each statement
    # divides out of it bounds the makespan. The objective's tie-break then
    # settles it at the fewest lanes reaching the best makespan, which is the
    # coincident tile count whenever the hardware has that many lanes.
    lane_split = model.NewIntVar(1, facts.lanes, "lane_split")
    for name, plan in plans.items():
        axes = facts.axes_of[name]
        volume_ub = math.prod(facts.extents[b] for b in axes) or 1
        parallel = _product(
            model, [counts[b] for b in axes if b in facts.coincident],
            volume_ub, f"parallel_{name}",
        )
        serial = _product(
            model, [counts[b] for b in axes if b not in facts.coincident],
            volume_ub, f"serial_{name}",
        )
        model.AddDivisionEquality(plan.chunks, parallel + lane_split - 1, lane_split)
        rounds = _product(model, [plan.chunks, serial], volume_ub, f"rounds_{name}")
        span = model.NewIntVar(1, horizon, f"span_{name}")
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

    sizes = tuple(solver.Value(v) for v in tile)
    ring_decisions = {buf: solver.Value(var) for buf, var in ring.items()}
    decisions = {
        "status": solver.StatusName(status),
        "makespan": solver.Value(makespan),
        "tile": sizes,
        "tiles": tuple(solver.Value(v) for v in counts),
        "coincident": facts.coincident,
        "lanes": facts.lanes,
        "lane_split": solver.Value(lane_split),
        "capacity_bytes": facts.capacity,
        "footprint_bytes": {
            buf: _occupancy_bytes(occupancy, sizes) * ring_decisions[buf]
            for buf, occupancy in facts.held.items()
        },
        "statements": {
            name: {
                "atom": _picked_atom(solver, plan),
                "place": _place_of(facts, name),
                "tile_atoms": (
                    math.prod(solver.Value(v) for v in plan.mult.values())
                    if plan.mult else None
                ),
                "compute": solver.Value(plan.compute),
                "load": solver.Value(plan.load),
                "stage": solver.Value(plan.stage),
                "lane_rounds": solver.Value(plan.chunks),
                "start": solver.Value(plan.start),
                "end": solver.Value(plan.end),
            }
            for name, plan in plans.items()
        },
        "ring": ring_decisions,
    }
    return dataclasses.replace(
        tg,
        tree=tile_band(outermost_band(tg.tree), sizes),
        ring=ring_decisions,
        decisions=decisions,
    )


def _picked_atom(solver: cp_model.CpSolver, plan: _Plan) -> str | None:
    if not plan.candidates:
        return None
    chosen = next(i for i in range(len(plan.candidates)) if solver.Value(plan.pick[i]))
    return plan.candidates[chosen].atom.op.name


def _place_of(facts: _Facts, name: str) -> str:
    """Placement is derived, never solved: the band members isl marked
    coincident that this statement varies in are the ones spread over
    lanes."""
    members = [b for b in facts.axes_of[name] if b in facts.coincident]
    if not members:
        return "serial"
    return "coincident[" + ",".join(str(b) for b in members) + "]"


__all__ = ["SolveResourcesError", "solve_resources"]
