"""When the modeled work runs, given a fixed number of parallel units.

Calls are first grouped into execution units: consecutive work that stays in the
same local storage at the same parallel extent is one launch, because nothing
between those calls forces a round trip. Each unit is then issued in waves of at
most the parallel capacity, and the waves are placed subject to the dependencies
between units.

The result is a plan. It says what the model believes about ordering and
occupancy, and it must not be read as a statement about lowering or about
measured performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ortools.sat.python import cp_model

from tilefoundry.ir.core import Call, get_metadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import Type
from tilefoundry.ir.types.shard import ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.target.amx.target import AmxTarget
from tilefoundry.target.cuda.target import CudaTarget
from tilefoundry.target.facts import TARGET_FACTS

from .errors import AnalysisError
from .facts import ParallelCapacityFacts
from .metadata import RooflineMetadata, TimelineMetadata
from .registry import register_analysis
from .walk import (
    attach,
    describe,
    postorder,
    reachable_functions,
    tensor_types,
    topology_extent,
)

SELECTOR = "timeline"

_SOLVE_SECONDS = 5.0


@dataclass
class _Unit:
    """One launch: the calls it issues, what it waits for, and how wide it is."""

    calls: list[Call]
    predecessors: set[int]
    extent: int
    duration_ns: int


def _is_local(type_: Type) -> bool:
    """Whether every tensor leaf of *type_* stays inside one parallel unit."""
    tensors = tensor_types(type_)
    return bool(tensors) and all(
        tensor.storage in {StorageKind.RMEM, StorageKind.SMEM} for tensor in tensors
    )


def _fusable(producer: Type, consumer: Type) -> bool:
    """Whether a value can be handed over without leaving the unit.

    Both sides must live in the same local storage and on the same mesh. Same
    storage alone is not enough: two values in registers on different meshes are
    held by different sets of threads, so passing one to the other is a
    redistribution rather than a read.
    """
    if not _is_local(producer) or not _is_local(consumer):
        return False
    producer_tensors = tensor_types(producer)
    consumer_tensors = tensor_types(consumer)
    producer_storages = {tensor.storage for tensor in producer_tensors}
    if len(producer_storages) != 1:
        return False
    if producer_storages != {tensor.storage for tensor in consumer_tensors}:
        return False
    producer_meshes = {
        tensor.layout.mesh
        for tensor in producer_tensors
        if isinstance(tensor.layout, ShardLayout)
    }
    consumer_meshes = {
        tensor.layout.mesh
        for tensor in consumer_tensors
        if isinstance(tensor.layout, ShardLayout)
    }
    return bool(producer_meshes) and producer_meshes == consumer_meshes


def _extent(call: Call, topologies: tuple[Topology, ...], name: str) -> int:
    """How many parallel units *call* occupies.

    The output decides when it says anything, then a single agreeing input, then
    the function's own declaration. A call that none of them describe occupies
    one unit: that is the smallest launch, not a guess at a larger one.
    """
    output = topology_extent(call.type, name)
    if output is not None:
        return output
    inputs = {
        value for arg in call.args if (value := topology_extent(arg.type, name))
    }
    if len(inputs) == 1:
        return next(iter(inputs))
    declared = {
        topology.size
        for topology in topologies
        if topology.name == name and isinstance(topology.size, int)
    }
    return next(iter(declared)) if len(declared) == 1 else 1


def _units(
    fn: Function,
    durations: dict[int, int],
    topologies: tuple[Topology, ...],
    topology: str,
) -> dict[int, _Unit]:
    """Group *fn*'s calls into execution units by fusable local placement.

    A reshard is never fused across: it exists precisely to move data between
    placements, so the unit must end there.
    """
    calls = [expr for expr in postorder(fn.body) if isinstance(expr, Call)]
    call_by_id = {id(call): call for call in calls}
    parent = {id(call): id(call) for call in calls}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for consumer in calls:
        if isinstance(consumer.target, Reshard):
            continue
        for arg in consumer.args:
            producer = call_by_id.get(id(arg))
            if producer is None or isinstance(producer.target, Reshard):
                continue
            if _fusable(producer.type, consumer.type) and _extent(
                producer, topologies, topology
            ) == _extent(consumer, topologies, topology):
                left, right = find(id(producer)), find(id(consumer))
                if left != right:
                    parent[right] = left

    index_of: dict[int, int] = {}
    for call in calls:
        root = find(id(call))
        if root not in index_of:
            index_of[root] = len(index_of)
    units = {index: _Unit([], set(), 1, 0) for index in index_of.values()}
    unit_of: dict[int, int] = {}
    for call in calls:
        unit_id = index_of[find(id(call))]
        unit_of[id(call)] = unit_id
        unit = units[unit_id]
        unit.calls.append(call)
        unit.extent = max(unit.extent, _extent(call, topologies, topology))
        unit.duration_ns += durations[id(call)]
    for consumer in calls:
        consumer_unit = unit_of[id(consumer)]
        for arg in consumer.args:
            producer_unit = unit_of.get(id(arg))
            if producer_unit is not None and producer_unit != consumer_unit:
                units[consumer_unit].predecessors.add(producer_unit)
    return units


def _wave_plan(unit: _Unit, capacity: int) -> list[tuple[int, int]]:
    """Split one unit into ``(demand, duration)`` waves of at most *capacity*.

    The unit's own duration is divided across its waves in proportion to how
    much of the unit each wave issues, and the last wave absorbs the remainder
    so the parts still sum to the whole.
    """
    demands: list[int] = []
    remaining = unit.extent
    while remaining > 0:
        demands.append(min(capacity, remaining))
        remaining -= demands[-1]
    if not demands:
        demands = [1]
    plan: list[tuple[int, int]] = []
    left = unit.duration_ns
    for index, demand in enumerate(demands):
        if index == len(demands) - 1:
            duration = left
        else:
            duration = math.ceil(unit.duration_ns * demand / unit.extent)
            left -= duration
        plan.append((demand, max(duration, 0)))
    return plan


def _solve(
    units: dict[int, _Unit], capacity: int
) -> tuple[int, int, dict[int, TimelineMetadata]]:
    """Place every unit's waves.

    Returns the makespan, the total number of waves issued, and one record per
    call keyed by identity. The wave total is counted here rather than derived
    from the records, because two distinct units can be placed identically and
    would otherwise collapse into one.
    """
    model = cp_model.CpModel()
    # Every wave is explicitly ordered inside its own unit, so the sum of the
    # unit durations is a valid finite upper bound even though independent units
    # will overlap once the cumulative resource constraint admits it.
    horizon = max(1, sum(max(1, unit.duration_ns) for unit in units.values()))
    waves_by_unit: dict[int, list[tuple[cp_model.IntVar, cp_model.IntVar]]] = {}
    intervals = []
    demands = []
    for unit_id, unit in units.items():
        waves: list[tuple[cp_model.IntVar, cp_model.IntVar]] = []
        for index, (demand, duration) in enumerate(_wave_plan(unit, capacity)):
            start = model.NewIntVar(0, horizon, f"u{unit_id}_w{index}_start")
            end = model.NewIntVar(0, horizon, f"u{unit_id}_w{index}_end")
            intervals.append(
                model.NewIntervalVar(start, duration, end, f"u{unit_id}_w{index}")
            )
            demands.append(demand)
            if waves:
                model.Add(start >= waves[-1][1])
            waves.append((start, end))
        waves_by_unit[unit_id] = waves
    for unit_id, unit in units.items():
        first_start = waves_by_unit[unit_id][0][0]
        for predecessor in unit.predecessors:
            model.Add(first_start >= waves_by_unit[predecessor][-1][1])
    model.AddCumulative(intervals, demands, capacity)
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan, [waves[-1][1] for waves in waves_by_unit.values()]
    )
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = _SOLVE_SECONDS
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise AnalysisError("the fixed-capacity timeline is infeasible")

    records: dict[int, TimelineMetadata] = {}
    for unit_id, unit in units.items():
        waves = waves_by_unit[unit_id]
        record = TimelineMetadata(
            grid_units=unit.extent,
            waves=len(waves),
            start_ns=solver.Value(waves[0][0]),
            end_ns=solver.Value(waves[-1][1]),
        )
        for call in unit.calls:
            records[id(call)] = record
    total_waves = sum(len(waves) for waves in waves_by_unit.values())
    return solver.Value(makespan), total_waves, records


def _durations(fn: Function) -> dict[int, int]:
    """Each call's modeled duration, from the bound the roofline family left."""
    result: dict[int, int] = {}
    for expr in postorder(fn.body):
        if not isinstance(expr, Call):
            continue
        bound = get_metadata(expr, RooflineMetadata)
        if bound is None:
            raise AnalysisError(
                f"{describe(expr)}: the timeline needs the roofline bound this "
                "call was never given"
            )
        result[id(expr)] = bound.theoretical_ns
    return result


def analyze_timeline(
    module: Module,
    function: Function,
    target: object,
    options: object | None = None,
) -> None:
    """Place every reachable Function's calls on the nominal timeline."""
    facts = TARGET_FACTS.project(target, ParallelCapacityFacts)
    capacity = facts.parallel_units
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise AnalysisError(
            "the timeline requires a positive parallel-unit capacity, got "
            f"{capacity!r}"
        )
    topologies = module.effective_topologies()
    for fn in reachable_functions(function):
        durations = _durations(fn)
        if not durations:
            attach(fn, TimelineMetadata(grid_units=0, waves=0, start_ns=0, end_ns=0))
            continue
        units = _units(fn, durations, topologies, facts.topology)
        makespan, total_waves, records = _solve(units, capacity)
        for expr in postorder(fn.body):
            record = records.get(id(expr))
            if record is not None:
                attach(expr, record)
        attach(
            fn,
            TimelineMetadata(
                grid_units=max(unit.extent for unit in units.values()),
                waves=total_waves,
                start_ns=0,
                end_ns=makespan,
            ),
        )


for _target_type in (CudaTarget, AmxTarget):
    register_analysis(
        _target_type,
        SELECTOR,
        requires=("roofline",),
        produces=(TimelineMetadata,),
    )(analyze_timeline)


__all__ = ["SELECTOR", "analyze_timeline"]
