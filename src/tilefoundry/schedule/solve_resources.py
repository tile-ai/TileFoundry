"""``solve_resources(tg, target=None, options=None) -> TileGraph`` --
CP-SAT fills every isl-statement's resource decisions (which MMA atom,
which lane it runs on, each buffer's ring/software-pipelining depth),
minimizes a coarse makespan, and returns ``tg`` with ``ring``/``decisions``
filled in -- refining a ``TileGraph`` whose ``tree`` is affine-only
(``kernel_schedule.py``, ``ring={}`` always) into one that also carries
resource decisions.

This is a from-scratch *statement*-granularity port of
``target.cuda.solver``'s HIR-granularity CP-SAT idiom -- ``pick`` BoolVar +
``AddExactlyOne`` to choose a candidate (``solver.py`` :183), an ``IntVar``
placement (``solver.py``'s ``offset`` :322-328), a ``makespan`` ``IntVar`` +
``Minimize`` (``solver.py`` :407/737) -- onto the isl statement domain
instead of ``solver.py``'s HIR-op/bucket domain.

Candidate atoms come from the target's own ``(Analysis, stage)`` service,
never from a target package this module imports: the CP-SAT model itself is
target-independent.

V1 simplifications:

* **atom choice** -- exactly ``solver.py``'s ``pick`` + ``AddExactlyOne``
  idiom, per statement, over the Analysis service's candidate list. A
  statement with no candidate (V1's CUDA catalogue is MatMul-only --
  anything else, e.g. RMSNorm, raises ``NotImplementedError``) is *not*
  given a pick var at all -- it gets a fixed nominal duration instead
  (see ``_default_duration_units``), so one uncovered op never aborts
  the whole solve.
* **place** -- one ``NewIntVar(0, lanes-1)`` per statement, ``lanes`` from
  ``target.device.sm_count`` when the target exposes one (a small fixed
  fallback otherwise). It is solved for and decoded, but -- unlike
  ``solver.py``'s ``offset`` -- V1 does not wire it into any packing
  constraint (no ``AddNoOverlap2D`` over lanes); see "makespan" below.
* **ring** -- one ``NewIntVar(_RING_MIN, _RING_MAX)`` per buffer written by
  some statement (``tg.writes``' ``OUT`` tuple names). V1 does not
  back-derive ``_RING_MAX`` from any real memory-capacity budget (that is
  ``solver.py``'s own ``AddCumulative``/``hbm_capacity_bytes`` accounting,
  out of scope here) and does not link ring depth to makespan/overlap --
  it is produced, not (yet) optimized over. The lower bound is 2, not 1,
  on purpose: ``render._ring_ref`` treats depth <= 1 as "no ring"
  (bare buffer name, no ``% N``), so a depth of 1 would be a CP-SAT
  decision silently invisible downstream; forcing >= 2 guarantees every
  produced ring decision is an observable one.
* **makespan** -- per-statement ``start``/``end`` IntVars
  (``end == start + duration``), coarsened dependence edges from
  ``tg.deps`` (instance-granularity) down to statement-granularity
  precedence pairs (``_statement_precedence_pairs``: "some instance of A
  must precede some instance of B") enforced as ``start[B] >= end[A]``,
  and ``makespan = max(all ends)``, minimized. This is a deps-chain
  makespan in place of a full
  ``AddNoOverlap2D`` time x resource pack: statements on an independent
  branch can still overlap (unconstrained relative to each other), while
  statements on a dependence chain serialize -- a real (if coarse)
  makespan shape, not a blind ``sum(duration)``. No lane/topology overlap
  constraint exists at all (that is ``solver.py``'s own hard problem).

Decisions land as plain ``TileGraph`` fields (``ring``, ``decisions``),
never an isl mark -- a Python dict is not safe payload for isl's own
C-managed tree nodes.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import isl
from ortools.sat.python import cp_model

from tilefoundry.analysis import Analysis, AtomFact
from tilefoundry.analysis.poly import TileGraph, TileUnit
from tilefoundry.target import Target, default_target, resolve_target

from . import ScheduleOptions

# ---------------------------------------------------------------------------
# V1 fixed constants -- see the module docstring's "V1 simplifications" for
# the reasoning behind each.
# ---------------------------------------------------------------------------

# ns -> integer CP-SAT duration units. An atom's own roofline estimate is
# floored at 1.0ns; a bare round(ns) would flatten every
# candidate to the same "1" and make atom choice cost-indifferent inside
# CP-SAT. Scaling by 1000 keeps 3 decimal digits of the roofline estimate.
_DURATION_SCALE = 1000

# Nominal per-statement-instance cost (ns), for a statement with no
# registered atom candidate (e.g. RMSNorm -- CUDA's V1 catalogue is
# MatMul-only, so it has no roofline cost model). Matches the same "~1ns"
# order of magnitude as the atom path's own floor; not a measured number.
_DEFAULT_DURATION_NS = 1.0

# Ring (software-pipelining stage count) domain. Lower bound 2 (not 1) so
# a produced ring decision is never the degenerate "no ring" depth (see
# module docstring). Upper bound is a small fixed V1 constant, not
# back-derived from any real shared-memory/register budget.
_RING_MIN = 2
_RING_MAX = 4

# `place` upper bound fallback, used only when `target` exposes no
# `device.sm_count` (e.g. a non-CUDA target).
_LANES_FALLBACK = 4


class SolveResourcesError(RuntimeError):
    """CP-SAT reported an infeasible or otherwise unusable resource model,
    or a ``tg``/``sched`` consistency precondition did not hold -- always
    raised with a specific, actionable message; V1 never silently ignores
    a bad solve."""


@dataclass(frozen=True)
class _StatementPlan:
    """One statement's CP-SAT variables, before solving.

    ``candidates``/``pick`` are both empty together (no atom candidate for
    this statement -- ``duration`` is then a plain fixed int, not a
    weighted-pick expression).
    """

    name: str
    candidates: tuple[AtomFact, ...]
    pick: tuple[cp_model.IntVar, ...]
    duration: object  # int, or a CP-SAT LinearExpr over `pick`
    place: cp_model.IntVar
    start: cp_model.IntVar
    end: cp_model.IntVar


# ---------------------------------------------------------------------------
# Per-statement facts: candidates / duration / precedence
# ---------------------------------------------------------------------------


def _analysis_service(target: Target, stage: str) -> Analysis:
    """The ``(Analysis, stage)`` service ``target`` binds -- the only route
    to a candidate atom, so a target that binds none is a solve-time error
    naming what is missing, not a silent no-candidate solve."""
    try:
        return target.service(Analysis, stage)
    except (TypeError, ValueError) as error:
        raise SolveResourcesError(
            f"solve_resources: target {target.name!r} binds no Analysis "
            f"service at stage {stage!r}: {error}"
        ) from error


def _candidates_for(unit: TileUnit, analysis: Analysis) -> list[AtomFact]:
    """``analysis.candidate_atoms``, made robust to the statements the
    target's catalogue was never meant to cover. CUDA's is MatMul-only; any
    other op (RMSNorm today) makes it raise ``NotImplementedError`` --
    caught here and downgraded to "no candidates" so one uncovered
    statement never aborts the whole resource solve."""
    try:
        return analysis.candidate_atoms(unit.op)
    except NotImplementedError:
        return []


def _duration_units(fact: AtomFact) -> int:
    return max(1, round(fact.duration * _DURATION_SCALE))


def _statement_volume(tg: TileGraph, stmt_name: str) -> int:
    """Product of ``stmt_name``'s own tiled-domain axis extents (the same
    ``dim_min_val``/``dim_max_val`` technique as
    ``render._statement_extents``) -- V1's only "how much work"
    signal for a statement with no atom candidate to size its default
    duration from."""
    sets: list["isl.set"] = []
    tg.domain.foreach_set(sets.append)
    for s in sets:
        if s.get_tuple_name() == stmt_name:
            rank = s.dim(isl.dim_type.SET)
            volume = 1
            for i in range(rank):
                lo = int(s.dim_min_val(i).num_si())
                hi = int(s.dim_max_val(i).num_si())
                volume *= hi - lo + 1
            return volume
    raise SolveResourcesError(
        f"solve_resources: no domain set found for statement {stmt_name!r} "
        "-- tg.domain and tg.units must come from the same extract() run"
    )


def _default_duration_units(tg: TileGraph, stmt_name: str) -> int:
    return _statement_volume(tg, stmt_name) * round(_DEFAULT_DURATION_NS * _DURATION_SCALE)


def _lane_count(target: Target) -> int:
    """V1's ``place`` upper bound: the target's parallel-unit count
    (``device.sm_count``, e.g. 132 on H200 SXM) when available, else a
    small fixed fallback for a target with no such device fact."""
    device = getattr(target, "device", None)
    sm_count = getattr(device, "sm_count", None)
    if isinstance(sm_count, int) and not isinstance(sm_count, bool) and sm_count > 0:
        return sm_count
    return _LANES_FALLBACK


def _written_buffers(tg: TileGraph) -> list[str]:
    """Every buffer name some statement writes (``tg.writes``' ``OUT``
    tuple names) -- the population @ring is decided over, per the task
    brief."""
    maps: list["isl.map"] = []
    tg.writes.foreach_map(maps.append)
    return sorted({m.get_tuple_name(isl.dim_type.OUT) for m in maps})


def _statement_precedence_pairs(tg: TileGraph) -> set[tuple[str, str]]:
    """Coarsen ``tg.deps`` (an instance-granularity must-dependence
    relation) down to statement-granularity precedence: ``(a, b)`` iff
    some instance of statement ``a`` must execute before some instance of
    statement ``b``. A same-statement dependence (e.g. MM's own k-carry)
    is dropped -- it constrains instance order *within* one statement,
    saying nothing about cross-statement order, and feeding it into
    ``start[a] >= end[a]`` would just make the model infeasible for any
    positive duration."""
    maps: list["isl.map"] = []
    tg.deps.foreach_map(maps.append)
    pairs: set[tuple[str, str]] = set()
    for m in maps:
        a = m.get_tuple_name(isl.dim_type.IN)
        b = m.get_tuple_name(isl.dim_type.OUT)
        if a != b:
            pairs.add((a, b))
    return pairs


# ---------------------------------------------------------------------------
# CP-SAT model
# ---------------------------------------------------------------------------


def _plan_statement(
    model: cp_model.CpModel,
    unit: TileUnit,
    candidates: list[AtomFact],
    default_units: int,
    lanes: int,
    horizon: int,
) -> _StatementPlan:
    """One statement's CP-SAT variables: atom ``pick`` (``solver.py``
    :183's ``AddExactlyOne`` idiom, skipped entirely when there are no
    candidates), ``place``, and a ``start``/``end`` pair whose difference
    is ``duration`` -- fixed to ``default_units`` when there is no atom
    to weigh choices by."""
    name = unit.name
    if candidates:
        pick = tuple(model.NewBoolVar(f"pick_{name}_{i}") for i in range(len(candidates)))
        model.AddExactlyOne(pick)
        units = [_duration_units(c) for c in candidates]
        duration = sum(p * u for p, u in zip(pick, units))
    else:
        pick = ()
        duration = default_units
    place = model.NewIntVar(0, lanes - 1, f"place_{name}")
    start = model.NewIntVar(0, horizon, f"start_{name}")
    end = model.NewIntVar(0, horizon, f"end_{name}")
    model.Add(end == start + duration)
    return _StatementPlan(
        name=name,
        candidates=tuple(candidates),
        pick=pick,
        duration=duration,
        place=place,
        start=start,
        end=end,
    )


def _horizon(tg: TileGraph, per_stmt_candidates: dict[str, list[AtomFact]]) -> int:
    """A safe static upper bound for every ``start``/``end``/``makespan``
    IntVar: the fully-serial (worst case) sum of each statement's own
    worst-case duration."""
    total = 0
    for unit in tg.units:
        candidates = per_stmt_candidates[unit.name]
        if candidates:
            total += max(_duration_units(c) for c in candidates)
        else:
            total += _default_duration_units(tg, unit.name)
    return max(1, total)


def solve_resources(
    tg: TileGraph,
    target: Target | str | None = None,
    options: ScheduleOptions | None = None,
    stage: str = "cta",
) -> TileGraph:
    """CP-SAT resource solve over ``tg`` (carrying the isl schedule tree
    already, from ``compute_schedule(tg)``): choose each statement's atom
    (when ``target``'s ``(Analysis, stage)`` service lists one), a lane
    placement, and a per-buffer ring pipeline depth, minimizing a coarse
    deps-chain makespan (see module docstring for exactly how simplified).
    Returns ``tg`` with ``ring``/``decisions`` filled in -- the exact
    ``{buffer_name: depth}`` shape ``render._ring_ref`` already knows how
    to consume, so ``emit_scaffold(solve_resources(compute_schedule(tg)))``
    renders ring-indexed (``% N``) buffer references end to end.

    Raises :class:`SolveResourcesError` if CP-SAT reports the model
    infeasible (or any other non-``OPTIMAL``/``FEASIBLE`` status) -- never
    silently returns a bogus solution.
    """
    if not tg.units:
        raise SolveResourcesError(
            "solve_resources: tg.units is empty -- nothing to solve resources for"
        )
    target = default_target() if target is None else resolve_target(target)
    options = options if options is not None else ScheduleOptions()

    analysis = _analysis_service(target, stage)
    lanes = _lane_count(target)
    per_stmt_candidates = {unit.name: _candidates_for(unit, analysis) for unit in tg.units}
    horizon = _horizon(tg, per_stmt_candidates)

    model = cp_model.CpModel()
    plans: dict[str, _StatementPlan] = {}
    for unit in tg.units:
        candidates = per_stmt_candidates[unit.name]
        default_units = 0 if candidates else _default_duration_units(tg, unit.name)
        plans[unit.name] = _plan_statement(model, unit, candidates, default_units, lanes, horizon)

    for a, b in _statement_precedence_pairs(tg):
        if a not in plans or b not in plans:
            raise SolveResourcesError(
                f"solve_resources: tg.deps references statement {a!r}->{b!r} "
                "not present in tg.units"
            )
        model.Add(plans[b].start >= plans[a].end)

    buffers = _written_buffers(tg)
    ring_vars = {buf: model.NewIntVar(_RING_MIN, _RING_MAX, f"ring_{buf}") for buf in buffers}

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, [plan.end for plan in plans.values()])
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = options.timeout_seconds
    solver.parameters.num_search_workers = options.workers
    solver.parameters.random_seed = options.random_seed
    status = solver.Solve(model)

    if status == cp_model.INFEASIBLE:
        raise SolveResourcesError(
            f"solve_resources: infeasible resource model for statements "
            f"{[u.name for u in tg.units]!r} on target {target.name!r}"
        )
    if status == cp_model.MODEL_INVALID:
        raise SolveResourcesError("solve_resources: OR-Tools reported an invalid model")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise SolveResourcesError(
            f"solve_resources: unexpected CP-SAT status "
            f"{solver.StatusName(status)!r}"
        )

    statement_decisions: dict[str, dict] = {}
    for name, plan in plans.items():
        atom_name = None
        if plan.candidates:
            chosen = next(i for i in range(len(plan.candidates)) if solver.Value(plan.pick[i]))
            atom_name = plan.candidates[chosen].atom.op.name
        statement_decisions[name] = {
            "atom": atom_name,
            "place": solver.Value(plan.place),
            "start": solver.Value(plan.start),
            "end": solver.Value(plan.end),
        }
    ring_decisions = {buf: solver.Value(var) for buf, var in ring_vars.items()}

    decisions = {
        "status": solver.StatusName(status),
        "makespan": solver.Value(makespan),
        "statements": statement_decisions,
        "ring": ring_decisions,
    }

    return dataclasses.replace(tg, ring=ring_decisions, decisions=decisions)


__all__ = ["SolveResourcesError", "solve_resources"]
