"""Place explicit host workspace using the protocol's completion dependencies."""

from dataclasses import dataclass
from itertools import combinations
from math import prod

import isl

from tilefoundry.ir import core, expr, types

from .allocation import AllocationSummaryMetadata, AllocationValue, solve_allocation
from .buffer_lifetime import BufferLifetime, interfere
from .communication import BufferBindings, BufferRegion, check_communication
from .errors import Error
from .memory import MemoryOptions
from .metadata import ValueLifetime
from .precision import Precision


@dataclass(frozen=True)
class WorkspacePlan:
    """A feasible arena layout, its constraints, and the binding checked at emission."""

    bindings: BufferBindings
    peak_bytes: int
    conflicts: tuple[tuple[str, str], ...]
    solver_status: str
    constraint_precision: Precision = Precision.EXACT


@dataclass(frozen=True)
class RegionBoundary:
    """External values and ordering obligations of a candidate atomic region."""

    operations: tuple[str, ...]
    inputs: tuple[BufferLifetime, ...]
    outputs: tuple[BufferLifetime, ...]
    prerequisites: tuple[str, ...]
    dependents: tuple[str, ...]


def overlap_candidates(report):
    """Return independent occurrence pairs from a checked execution, without costs."""
    if report.hazards:
        raise ValueError("overlap queries require an execution without buffer hazards")
    return tuple((a.name, b.name) for a, b in combinations(report.operations, 2)
                 if a.complete not in report.predecessors[b.read] and b.complete not in report.predecessors[a.read])


def region_boundary(report, operations) -> RegionBoundary:
    """Check contraction dependencies and expose the values a realization must preserve.

    This does not establish that a backend can fuse the selected operations or
    make distributed progress. It preserves the observable boundary for that check.
    """
    if report.hazards:
        raise ValueError("region queries require an execution without buffer hazards")
    names = tuple(operations)
    by_name = {operation.name: operation for operation in report.operations}
    if not names or len(set(names)) != len(names) or set(names) - by_name.keys():
        raise ValueError("a region must name unique execution occurrences")
    selected = tuple(operation for operation in report.operations if operation.name in names)
    outside = tuple(operation for operation in report.operations if operation.name not in names)
    before, after = set(), set()
    for other in outside:
        earlier = any(op.complete in report.predecessors[other.read] for op in selected)
        later = any(other.complete in report.predecessors[op.read] for op in selected)
        if earlier and later:
            raise ValueError(f"region is not dependency-convex: {other.name} must execute between its members")
        if earlier:
            after.add(other.name)
        if later:
            before.add(other.name)
    completions = {op.complete for op in selected}
    reads = {op.read for op in selected}
    for event, frontier in report.exported_frontiers:
        covered = set(frontier).union(*(report.predecessors[node] for node in frontier))
        if completions & covered and not completions <= covered:
            raise ValueError(f"region has an externally visible intermediate completion: {event}")
    inputs = tuple(value for value in report.lifetimes if value.producer.operation not in completions
                   and any(use.operation in reads for use in value.consumers))
    outputs = tuple(value for value in report.lifetimes if value.producer.operation in completions
                    and (value.value in report.live_out or any(use.operation not in reads for use in value.consumers)))
    return RegionBoundary(tuple(op.name for op in selected), inputs, outputs, tuple(sorted(before)), tuple(sorted(after)))


def forbidden_offsets(first, second):
    """Relative root offsets that could overlap two byte sets."""
    differences = first.flat_product(second).apply(isl.map("{ [a,b] -> [d] : d = a-b }"))
    parts = differences.get_basic_set_list()
    intervals, precision = [], Precision.EXACT
    for index in range(parts.size()):
        part = parts.get_at(index).to_set()
        low, high = part.dim_min_val(0).get_num_si(), part.dim_max_val(0).get_num_si()
        if not part.is_equal(isl.set(f"{{ [d] : {low} <= d <= {high} }}")):
            precision = Precision.UPPER_BOUND
        intervals.append((low, high))
    return intervals, precision


def plan_workspace(function, parameters, *, module=None, capacity_bytes=None, options=None) -> WorkspacePlan:
    """Assign explicitly disposable scratch parameters to one shared arena.

    Caller inputs and outputs are never implicitly reclaimed. Scratch must be
    written before being read. The existing solver consumes completion-derived
    conflicts; its result is rechecked by the same physical lifetime checker.
    Attach the returned bindings to the host function before code generation.
    """
    names = tuple(parameters)
    params = {param.name: param for param in function.params}
    if not names or len(names) != len(set(names)) or any(name not in params for name in names):
        raise ValueError("workspace requires unique declared parameter names")
    outputs = function.params[-function.output_count:] if function.output_count else ()
    if any(name in {param.name for param in outputs} for name in names):
        raise ValueError("caller outputs cannot be reclaimed as workspace")
    held = [params[name].type for name in names]
    if any(not isinstance(ty, types.TensorType) or ty.storage not in {expr.StorageKind.GMEM, expr.StorageKind.HOST}
           for ty in held) or len({ty.storage for ty in held}) != 1:
        raise ValueError("workspace parameters must share one host or device memory space")
    report = check_communication(function, regions={}, module=module, uninitialized=names)
    if any(hazard.kind == "uninitialized_read" for hazard in report.hazards):
        raise ValueError(f"workspace cannot require an initial input: {report.hazards}")
    if report.hazards:
        raise ValueError(f"cannot allocate an unsafe execution: {report.hazards}")
    lifetimes = {name: tuple(value for value in report.lifetimes if value.allocation == name
                            and (value.producer.operation != "entry" or value.consumers)) for name in names}
    for name, values in lifetimes.items():
        if not values or any(value.producer.operation == "entry" for value in values):
            raise ValueError(f"workspace {name!r} must be written before use and cannot hold an initial input")
    conflicts, precision = {}, Precision.EXACT
    for first, second in combinations(names, 2):
        pairs = [(a.region, b.region) for a in lifetimes[first] for b in lifetimes[second]
                 if interfere(a, b, predecessors=report.predecessors)]
        pairs += [(a.region.relation.range(), b.region.relation.range()) for operation in report.operations
                  for a in operation.effects if a.parameter == first
                  for b in operation.effects if b.parameter == second and (a.mode | b.mode) & expr.MemoryEffect.WRITE]
        intervals = []
        for a, b in pairs:
            forbidden, bound = forbidden_offsets(a, b)
            intervals.extend(forbidden)
            precision = precision.join(bound)
        if intervals:
            conflicts[first, second] = tuple(sorted(set(intervals)))
    values = []
    memory_level = str(held[0].storage)
    for name in names:
        accesses = lifetimes[name]
        size = prod(params[name].type.shape) * params[name].type.dtype.bit_width // 8
        lifetime = ValueLifetime(core.SSAValue(params[name]), min(v.producer.start for v in accesses),
                                 max(v.end for v in accesses), memory_level=memory_level, bytes=size)
        values.append(AllocationValue(lifetime))
    summary = AllocationSummaryMetadata(tuple(values), windows=tuple(v.lifetime for v in values),
                                        sizes={v: v.bytes for v in values})
    try:
        result = solve_allocation(
            summary, None, options or MemoryOptions(), memory_level=memory_level,
            completion_conflicts={(core.SSAValue(params[a]), core.SSAValue(params[b])): intervals
                                  for (a, b), intervals in conflicts.items()},
            capacity_bytes=capacity_bytes,
        )
    except Error as error:
        if precision is not Precision.EXACT:
            raise Error(f"conservative workspace constraints found no placement; {error}") from error
        raise
    regions = tuple((value.value.expr.name, BufferRegion(f"{function.name}.workspace", offset))
                    for value, offset in result.offsets.items())
    checked = check_communication(function, regions=dict(regions), module=module, uninitialized=names)
    if checked.hazards:
        raise ValueError(f"allocated workspace failed completion checking: {checked.hazards}")
    return WorkspacePlan(BufferBindings(regions, names), result.peak_bytes, tuple(sorted(conflicts)), result.solver_status, precision)


__all__ = ["RegionBoundary", "WorkspacePlan", "overlap_candidates", "plan_workspace", "region_boundary"]
