from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from ortools.sat.python import cp_model

from tilefoundry.ir.core import (
    Call,
    Constant,
    Expr,
    IRMetadata,
    SourceSpanMetadata,
    Tuple,
    binding_name,
    get_metadata,
)
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.hir.grid_region import GridRegionExpr
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.types import TensorType, TupleType, Type, callable_type_for
from tilefoundry.ir.types.shard import ShardLayout
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.schedule.constraints import ScheduleConstraintMetadata
from tilefoundry.target import CudaTarget, Target, default_target
from tilefoundry.target.cuda.cost import tensor_bytes
from tilefoundry.visitor_registry import cost_evaluator_registry
from tilefoundry.visitor_registry.contexts import Cost, CostContext, TypeInferContext
from tilefoundry.visitor_registry.visitors import TypeInferVisitor

from .metadata import (
    FootprintMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TrafficBytes,
)


class AnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class AnalysisOptions:
    roofline: bool = True
    footprint: bool = True
    timeline: bool = True

    @property
    def metadata_types(self) -> tuple[type[IRMetadata], ...]:
        result: list[type[IRMetadata]] = []
        if self.roofline:
            result.append(RooflineMetadata)
        if self.footprint:
            result.append(FootprintMetadata)
        if self.timeline:
            result.append(TimelineMetadata)
        return tuple(result)


@dataclass(frozen=True)
class AnalysisResult:
    ir: Module | Function
    summary_lines: tuple[str, ...]
    metadata_types: tuple[type[IRMetadata], ...]


def _children(expr: Expr) -> tuple[Expr, ...]:
    match expr:
        case Call(args=args):
            return args
        case Tuple(elements=elements):
            return elements
        case GridRegionExpr(init_args=init_args, body=body, yield_values=yield_values):
            return (*init_args, body, *yield_values)
        case _:
            return ()


def _postorder(root: Expr | None) -> tuple[Expr, ...]:
    if root is None:
        return ()
    seen: set[int] = set()
    result: list[Expr] = []

    def visit(expr: Expr) -> None:
        if id(expr) in seen:
            return
        seen.add(id(expr))
        for child in _children(expr):
            visit(child)
        result.append(expr)

    visit(root)
    return tuple(result)


def _tensor_types(type_: Type) -> tuple[TensorType, ...]:
    match type_:
        case TensorType():
            return (type_,)
        case TupleType(fields=fields):
            return tuple(field for item in fields for field in _tensor_types(item))
        case _:
            return ()


def _storage_name(storage: StorageKind | None) -> str | None:
    return storage.name.lower() if storage is not None else None


def _bytes_by_storage(type_: Type) -> dict[str, int]:
    result: dict[str, int] = {}
    for tensor in _tensor_types(type_):
        if tensor.storage is StorageKind.UMAT:
            continue
        name = _storage_name(tensor.storage)
        if name is None:
            continue
        result[name] = result.get(name, 0) + tensor_bytes(tensor)
    return result


def _has_constraint(expr: Expr) -> bool:
    return get_metadata(expr, ScheduleConstraintMetadata) is not None


def _describe(expr: Expr) -> str:
    binding = binding_name(expr)
    span = get_metadata(expr, SourceSpanMetadata)
    prefix = f"{span.file}:{span.line}:{span.column}: " if span is not None else ""
    op = type(expr.target).__name__ if isinstance(expr, Call) else type(expr).__name__
    return f"{prefix}binding={binding or '<unnamed>'} op={op}"


def _validate_authored(functions: Iterable[Function]) -> None:
    for fn in functions:
        values = (*fn.params, *_postorder(fn.body))
        for expr in values:
            if _has_constraint(expr):
                raise AnalysisError(
                    f"{_describe(expr)}: authored analysis does not accept where(...); "
                    "write a concrete layout/storage with Tensor annotations or reshard"
                )
            if not isinstance(expr, Call):
                continue
            for tensor in _tensor_types(expr.type):
                if (
                    tensor.storage in {StorageKind.RMEM, StorageKind.SMEM}
                    and tensor.shape
                    and tensor.layout is None
                ):
                    raise AnalysisError(
                        f"{_describe(expr)}: distribution inference stopped with an "
                        "unresolved layout"
                    )
        if fn.body is not None:
            for tensor in _tensor_types(fn.body.type):
                if (
                    tensor.storage in {StorageKind.RMEM, StorageKind.SMEM}
                    and tensor.shape
                    and tensor.layout is None
                ):
                    raise AnalysisError(
                        f"function {fn.name!r} result: distribution inference stopped "
                        "with an unresolved layout"
                    )


def _infer_authored_types(
    functions: Iterable[Function], module: Module | None,
) -> None:
    """Re-derive every authored value type without candidate enumeration."""
    for fn in reversed(tuple(functions)):
        ctx = TypeInferContext(module=module)
        worklist = list(_postorder(fn.body))
        for expr in worklist:
            computed = TypeInferVisitor(ctx).visit(expr)
            if computed != expr.type:
                object.__setattr__(expr, "type", computed)
            ctx.cache[id(expr)] = computed
        if fn.body is not None and fn.return_type != fn.body.type:
            object.__setattr__(fn, "return_type", fn.body.type)
            object.__setattr__(fn, "type", callable_type_for(fn.params, fn.body.type))


def _replace_metadata_in_place(expr: Expr, value) -> None:
    updated = tuple(item for item in expr.metadata if type(item) is not type(value))
    object.__setattr__(expr, "metadata", (*updated, value))


def _cost(call: Call, module: Module | None) -> Cost:
    if isinstance(call.target, Function):
        return Cost({}, 0)
    fn = cost_evaluator_registry.lookup(type(call.target))
    if fn is None:
        raise AnalysisError(
            f"{_describe(call)}: no cost evaluator registered for "
            f"{type(call.target).__name__}"
        )
    return fn(call, CostContext(module=module))


def _scale_cost(cost: Cost, count: int) -> Cost:
    return Cost(
        {dtype: value * count for dtype, value in cost.flops.items()},
        cost.bytes * count,
    )


def _merge_traffic(
    destination: dict[str, TrafficBytes],
    traffic: tuple[tuple[str, TrafficBytes], ...],
) -> None:
    for level, value in traffic:
        current = destination.get(level, TrafficBytes())
        destination[level] = TrafficBytes(
            current.read_bytes + value.read_bytes,
            current.write_bytes + value.write_bytes,
        )


def _traffic_tuple(values: dict[str, TrafficBytes]) -> tuple[tuple[str, TrafficBytes], ...]:
    return tuple(sorted(values.items()))


def _traffic(call: Call, cost: Cost) -> tuple[tuple[str, TrafficBytes], ...]:
    if cost.bytes == 0:
        return ()
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}
    for arg in call.args:
        for name, value in _bytes_by_storage(arg.type).items():
            reads[name] = reads.get(name, 0) + value
    for name, value in _bytes_by_storage(call.type).items():
        writes[name] = writes.get(name, 0) + value
    levels = sorted(set(reads) | set(writes))
    return tuple(
        (name, TrafficBytes(reads.get(name, 0), writes.get(name, 0)))
        for name in levels
    )


def _target_for(ir: Module | Function) -> Target:
    if isinstance(ir, Function) and ir.target is not None:
        return ir.target
    if isinstance(ir, Module):
        entry = ir.entry_function()
        if isinstance(entry, Function) and entry.target is not None:
            return entry.target
        configured = ir.metadata.get("target")
        if isinstance(configured, Target):
            return configured
    return default_target()


def _bound_ns(cost: Cost, traffic, target: Target) -> int:
    if not isinstance(target, CudaTarget):
        return 0
    compute_ns = 0
    for dtype, flops in cost.flops.items():
        if not flops:
            continue
        try:
            peak = target.device.peak_for(dtype)
        except ValueError:
            continue
        compute_ns += math.ceil(flops * 1_000_000_000 / peak)
    gmem = next((value for name, value in traffic if name == "gmem"), None)
    memory_bytes = 0 if gmem is None else gmem.read_bytes + gmem.write_bytes
    memory_ns = (
        math.ceil(
            memory_bytes * 1_000_000_000
            / target.device.hbm_bandwidth_bytes_per_second
        )
        if memory_bytes
        else 0
    )
    return max(compute_ns, memory_ns, 1 if cost.flops or cost.bytes else 0)


def _cta_count_from_type(type_: Type) -> int | None:
    counts: set[int] = set()
    for tensor in _tensor_types(type_):
        if not isinstance(tensor.layout, ShardLayout):
            continue
        for topology in tensor.layout.mesh.topologies:
            if topology.name == "cta":
                if not isinstance(topology.size, int) or topology.size <= 0:
                    raise AnalysisError(
                        "timeline requires a positive static CTA topology extent"
                    )
                counts.add(topology.size)
    if len(counts) > 1:
        raise AnalysisError(f"one value references conflicting CTA extents {sorted(counts)}")
    return next(iter(counts), None)


def _execution_domain_from_type(type_: Type) -> dict[str, int] | None:
    domains: set[tuple[tuple[str, int], ...]] = set()
    for tensor in _tensor_types(type_):
        if not isinstance(tensor.layout, ShardLayout):
            continue
        domain: dict[str, int] = {}
        for topology in tensor.layout.mesh.topologies:
            if not isinstance(topology.size, int) or topology.size <= 0:
                raise AnalysisError(
                    "roofline requires positive static execution topology extents"
                )
            previous = domain.get(topology.name)
            if previous is not None and previous != topology.size:
                raise AnalysisError(
                    f"one Mesh declares conflicting {topology.name!r} extents "
                    f"{previous} and {topology.size}"
                )
            domain[topology.name] = topology.size
        domains.add(tuple(sorted(domain.items())))
    if len(domains) > 1:
        raise AnalysisError(
            f"one value references conflicting execution domains {sorted(domains)}"
        )
    return dict(next(iter(domains))) if domains else None


def _cta_count(call: Call, fn: Function) -> int:
    output = _cta_count_from_type(call.type)
    if output is not None:
        return output
    inputs = {value for arg in call.args if (value := _cta_count_from_type(arg.type))}
    if len(inputs) == 1:
        return next(iter(inputs))
    if isinstance(call.target, Function):
        target_declared = {
            topology.size
            for topology in call.target.topologies
            if topology.name == "cta" and isinstance(topology.size, int)
        }
        if len(target_declared) == 1:
            return next(iter(target_declared))
    declared = {
        topology.size
        for topology in fn.topologies
        if topology.name == "cta" and isinstance(topology.size, int)
    }
    return next(iter(declared)) if len(declared) == 1 else 1


def _execution_count(call: Call, fn: Function) -> int:
    domain = _execution_domain_from_type(call.type)
    inputs = {
        tuple(sorted(value.items()))
        for arg in call.args
        if (value := _execution_domain_from_type(arg.type)) is not None
    }
    if domain is None and len(inputs) > 1:
        raise AnalysisError(
            f"{_describe(call)}: inputs reference conflicting execution domains "
            f"{sorted(inputs)}"
        )
    if domain is None:
        domain = dict(next(iter(inputs))) if inputs else {}
    for topology in fn.topologies:
        if not isinstance(topology.size, int) or topology.size <= 0:
            raise AnalysisError(
                f"function {fn.name!r}: roofline requires positive static "
                "execution topology extents"
            )
        previous = domain.get(topology.name)
        if previous is not None and previous != topology.size:
            raise AnalysisError(
                f"{_describe(call)}: value Mesh declares {topology.name}={previous}, "
                f"but function {fn.name!r} declares {topology.name}={topology.size}"
            )
        domain[topology.name] = topology.size
    return math.prod(domain.values())


def _is_local(type_: Type) -> bool:
    tensors = _tensor_types(type_)
    return bool(tensors) and all(
        tensor.storage in {StorageKind.RMEM, StorageKind.SMEM} for tensor in tensors
    )


def _local_placement_compatible(producer: Type, consumer: Type) -> bool:
    if not _is_local(producer) or not _is_local(consumer):
        return False
    producer_tensors = _tensor_types(producer)
    consumer_tensors = _tensor_types(consumer)
    producer_storages = {tensor.storage for tensor in producer_tensors}
    consumer_storages = {tensor.storage for tensor in consumer_tensors}
    if len(producer_storages) != 1 or producer_storages != consumer_storages:
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


@dataclass
class _Unit:
    calls: list[Call]
    predecessors: set[int]
    grid_ctas: int
    duration_ns: int


def _timeline_for_function(
    fn: Function,
    costs: dict[int, tuple[Cost, tuple[tuple[str, TrafficBytes], ...], int]],
    capacity: int,
) -> tuple[int, dict[int, TimelineMetadata]]:
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
        raise AnalysisError(
            "fixed CTA timeline requires a positive compiler parallel CTA capacity"
        )
    calls = [expr for expr in _postorder(fn.body) if isinstance(expr, Call)]
    call_by_id = {id(call): call for call in calls}
    parent = {id(call): id(call) for call in calls}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for consumer in calls:
        for arg in consumer.args:
            if id(arg) not in call_by_id or isinstance(consumer.target, Reshard):
                continue
            producer = call_by_id[id(arg)]
            if isinstance(producer.target, Reshard):
                continue
            if (
                _local_placement_compatible(producer.type, consumer.type)
                and _cta_count(producer, fn) == _cta_count(consumer, fn)
            ):
                union(id(producer), id(consumer))

    root_to_index: dict[int, int] = {}
    for call in calls:
        root = find(id(call))
        if root not in root_to_index:
            root_to_index[root] = len(root_to_index)
    units = {
        index: _Unit([], set(), 1, 0) for index in root_to_index.values()
    }
    call_unit: dict[int, int] = {}
    for call in calls:
        unit_id = root_to_index[find(id(call))]
        call_unit[id(call)] = unit_id
        unit = units[unit_id]
        unit.calls.append(call)
        unit.grid_ctas = max(unit.grid_ctas, _cta_count(call, fn))
        unit.duration_ns += costs[id(call)][2]
    for consumer in calls:
        consumer_unit = call_unit[id(consumer)]
        for arg in consumer.args:
            producer_unit = call_unit.get(id(arg))
            if producer_unit is not None and producer_unit != consumer_unit:
                units[consumer_unit].predecessors.add(producer_unit)

    model = cp_model.CpModel()
    # Every wave below is explicitly ordered inside its execution unit.  The
    # sum is therefore a valid finite upper bound even when independent units
    # later overlap through the cumulative resource constraint.
    horizon = max(1, sum(max(1, unit.duration_ns) for unit in units.values()))
    unit_waves: dict[int, list[tuple[cp_model.IntVar, cp_model.IntVar, int]]] = {}
    intervals = []
    demands = []
    for unit_id, unit in units.items():
        remaining = unit.grid_ctas
        wave_demands = []
        while remaining > 0:
            wave_demands.append(min(capacity, remaining))
            remaining -= wave_demands[-1]
        if not wave_demands:
            wave_demands = [1]
        wave_durations = []
        remaining_duration = unit.duration_ns
        for index, demand in enumerate(wave_demands):
            if index == len(wave_demands) - 1:
                duration = remaining_duration
            else:
                duration = math.ceil(unit.duration_ns * demand / unit.grid_ctas)
                remaining_duration -= duration
            wave_durations.append(max(duration, 0))
        waves = []
        for wave_index, (demand, duration) in enumerate(zip(wave_demands, wave_durations)):
            start = model.NewIntVar(0, horizon, f"u{unit_id}_w{wave_index}_start")
            end = model.NewIntVar(0, horizon, f"u{unit_id}_w{wave_index}_end")
            interval = model.NewIntervalVar(
                start, duration, end, f"u{unit_id}_w{wave_index}"
            )
            intervals.append(interval)
            demands.append(demand)
            waves.append((start, end, demand))
            if wave_index:
                model.Add(start >= waves[wave_index - 1][1])
        unit_waves[unit_id] = waves
    for unit_id, unit in units.items():
        first_start = unit_waves[unit_id][0][0]
        for predecessor in unit.predecessors:
            model.Add(first_start >= unit_waves[predecessor][-1][1])
    model.AddCumulative(intervals, demands, capacity)
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan, [waves[-1][1] for waves in unit_waves.values()]
    )
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise AnalysisError("fixed CTA timeline is infeasible")

    metadata: dict[int, TimelineMetadata] = {}
    for unit_id, unit in units.items():
        waves = unit_waves[unit_id]
        value = TimelineMetadata(
            grid_ctas=unit.grid_ctas,
            waves=len(waves),
            start_ns=solver.Value(waves[0][0]),
            end_ns=solver.Value(waves[-1][1]),
        )
        for call in unit.calls:
            metadata[id(call)] = value
    return solver.Value(makespan), metadata


def _footprint_for_function(fn: Function) -> tuple[dict[int, FootprintMetadata], dict[str, int]]:
    order = [expr for expr in _postorder(fn.body) if isinstance(expr, (Call, Constant))]
    positions = {id(expr): index for index, expr in enumerate(order)}
    last_use = {id(expr): positions[id(expr)] for expr in order}
    for consumer in order:
        for child in _children(consumer):
            if id(child) in last_use:
                last_use[id(child)] = max(last_use[id(child)], positions[id(consumer)])
    if fn.body is not None and id(fn.body) in last_use:
        last_use[id(fn.body)] = len(order) - 1
    allocations: dict[int, dict[str, int]] = {}
    for expr in order:
        if isinstance(expr, Call) and isinstance(expr.target, (Reshape, Transpose)):
            allocations[id(expr)] = {}
        else:
            allocations[id(expr)] = _bytes_by_storage(expr.type)
    result: dict[int, FootprintMetadata] = {}
    peaks: dict[str, int] = {}
    for index, expr in enumerate(order):
        live: dict[str, int] = {}
        for value in order:
            born = positions[id(value)]
            if born <= index <= last_use[id(value)]:
                for level, amount in allocations[id(value)].items():
                    live[level] = live.get(level, 0) + amount
        for level, amount in live.items():
            peaks[level] = max(peaks.get(level, 0), amount)
        result[id(expr)] = FootprintMetadata(tuple(sorted(live.items())))
    return result, peaks


def _reachable_functions(ir: Module | Function) -> tuple[Function, ...]:
    if isinstance(ir, Function):
        roots = (ir,)
    else:
        entry = ir.entry_function()
        if not isinstance(entry, Function):
            raise AnalysisError("analyze currently accepts HIR functions only")
        roots = (entry,)
    result: list[Function] = []
    seen: set[int] = set()

    def visit(fn: Function) -> None:
        if id(fn) in seen:
            return
        seen.add(id(fn))
        result.append(fn)
        for expr in _postorder(fn.body):
            if isinstance(expr, Call) and isinstance(expr.target, Function):
                visit(expr.target)

    for root in roots:
        visit(root)
    return tuple(result)


@dataclass(frozen=True)
class _FunctionMetrics:
    cost: Cost
    traffic: tuple[tuple[str, TrafficBytes], ...]
    roofline_ns: int
    makespan_ns: int
    peak_footprint: tuple[tuple[str, int], ...]


def _sum_costs(costs: Iterable[Cost]) -> Cost:
    flops = {}
    byte_count = 0
    for cost in costs:
        byte_count += cost.bytes
        for dtype, value in cost.flops.items():
            flops[dtype] = flops.get(dtype, 0) + value
    return Cost(flops, byte_count)


def analyze(
    ir: Module | Function,
    *,
    options: AnalysisOptions | None = None,
) -> AnalysisResult:
    options = options or AnalysisOptions()
    functions = _reachable_functions(ir)
    module = ir if isinstance(ir, Module) else None
    _infer_authored_types(functions, module)
    _validate_authored(functions)
    target = _target_for(ir)
    # This is intentionally a compiler policy, not CUDA's grid limit nor the
    # hardware resident-CTA maximum.  The initial H200 policy is one active
    # CTA per SM; a target can expose a tighter policy without changing IR.
    capacity = (
        getattr(target.device, "compiler_policy_max_parallel_ctas", target.device.sm_count)
        if isinstance(target, CudaTarget)
        else 1
    )
    metrics_by_function: dict[int, _FunctionMetrics] = {}

    for fn in reversed(functions):
        costs: dict[int, tuple[Cost, tuple[tuple[str, TrafficBytes], ...], int]] = {}
        for expr in _postorder(fn.body):
            if not isinstance(expr, Call):
                continue
            if isinstance(expr.target, Function):
                child = metrics_by_function.get(id(expr.target))
                if child is None:
                    raise AnalysisError(
                        f"{_describe(expr)}: recursive or unresolved Function call graph"
                    )
                cost = child.cost
                traffic = child.traffic
                roofline_bound = child.roofline_ns
                duration = child.makespan_ns if options.timeline else roofline_bound
            else:
                local_cost = _cost(expr, module)
                cost = _scale_cost(local_cost, _execution_count(expr, fn))
                traffic = _traffic(expr, cost)
                roofline_bound = _bound_ns(cost, traffic, target)
                duration = roofline_bound
            costs[id(expr)] = (cost, traffic, duration)
            if options.roofline:
                metadata = RooflineMetadata(
                    tuple(sorted((dtype.name, value) for dtype, value in cost.flops.items())),
                    traffic,
                    roofline_bound,
                )
                _replace_metadata_in_place(expr, metadata)
        local_peaks: dict[str, int] = {}
        if options.footprint:
            footprint, local_peaks = _footprint_for_function(fn)
            for expr in _postorder(fn.body):
                value = footprint.get(id(expr))
                if value is not None:
                    if isinstance(expr, Call) and isinstance(expr.target, Function):
                        child = metrics_by_function[id(expr.target)]
                        live = dict(value.live_bytes)
                        for level, amount in _bytes_by_storage(expr.type).items():
                            remaining = live.get(level, 0) - amount
                            if remaining > 0:
                                live[level] = remaining
                            else:
                                live.pop(level, None)
                        for level, amount in child.peak_footprint:
                            live[level] = live.get(level, 0) + amount
                        value = FootprintMetadata(tuple(sorted(live.items())))
                        for level, amount in live.items():
                            local_peaks[level] = max(local_peaks.get(level, 0), amount)
                    _replace_metadata_in_place(expr, value)
        makespan = 0
        if options.timeline and costs:
            makespan, timeline = _timeline_for_function(fn, costs, capacity)
            for expr in _postorder(fn.body):
                value = timeline.get(id(expr))
                if value is not None:
                    _replace_metadata_in_place(expr, value)

        total_cost = _sum_costs(value[0] for value in costs.values())
        total_traffic: dict[str, TrafficBytes] = {}
        for _, traffic, _ in costs.values():
            _merge_traffic(total_traffic, traffic)
        traffic_tuple = _traffic_tuple(total_traffic)
        metrics_by_function[id(fn)] = _FunctionMetrics(
            cost=total_cost,
            traffic=traffic_tuple,
            roofline_ns=_bound_ns(total_cost, traffic_tuple, target),
            makespan_ns=makespan,
            peak_footprint=tuple(sorted(local_peaks.items())),
        )

    root = ir.entry_function() if isinstance(ir, Module) else ir
    if not isinstance(root, Function):
        raise AnalysisError("analyze currently accepts HIR functions only")
    totals = metrics_by_function[id(root)]

    analysis_names = [
        name
        for name, enabled in (
            ("roofline", options.roofline),
            ("footprint", options.footprint),
            ("timeline", options.timeline),
        )
        if enabled
    ]
    summary = [
        f"analysis target={getattr(target, 'name', type(target).__name__)} "
        f"analyses={','.join(analysis_names)}",
    ]
    if options.roofline:
        summary.append(
            "flops "
            + (
                ", ".join(
                    f"{dtype.name}={value}"
                    for dtype, value in sorted(
                        totals.cost.flops.items(), key=lambda item: item[0].name
                    )
                )
                or "0"
            )
        )
        summary.append(
            "traffic "
            + (
                ", ".join(
                    f"{level}=r{value.read_bytes}/w{value.write_bytes}"
                    for level, value in totals.traffic
                )
                or "0"
            )
        )
    if options.footprint:
        summary.append(
            "peak-footprint "
            + (
                ", ".join(f"{level}={value}" for level, value in totals.peak_footprint)
                or "0"
            )
        )
    if options.timeline:
        summary.append(f"theoretical-makespan={totals.makespan_ns}ns")
    return AnalysisResult(ir, tuple(summary), options.metadata_types)


__all__ = [
    "AnalysisError",
    "AnalysisOptions",
    "AnalysisResult",
    "analyze",
]
