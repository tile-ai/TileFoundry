"""Private CP-SAT solve and decoded CTA planning result."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

from ortools.sat.python import cp_model

from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.types import TensorType, Type
from tilefoundry.ir.types.shard import ComposedLayout, ShardLayout
from tilefoundry.schedule import ScheduleOptions

from .cost import tensor_bytes
from .debug import write_debug_dumps
from .planner import (
    OpCandidate,
    PlanningProblem,
    RegionInfo,
)

_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class ExecutionInterval:
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class PlanningSolution:
    status: Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]
    selected_candidate_ids: tuple[int, ...]
    selected_bucket_ids: tuple[int, ...]
    candidate_intervals_ns: tuple[tuple[int, ExecutionInterval], ...]
    bucket_offsets: tuple[tuple[int, int], ...]
    makespan_ns: int
    best_bound_ns: int
    gap: float


@dataclass
class _CpModelState:
    model: cp_model.CpModel
    pick_candidates: dict[int, cp_model.IntVar]
    pick_buckets: dict[int, cp_model.IntVar]
    starts: dict[int, cp_model.IntVar]
    ends: dict[int, cp_model.IntVar]
    ready: dict[int, cp_model.IntVar]
    offsets: dict[int, cp_model.IntVar]
    makespan: cp_model.IntVar
    horizon_ns: int


def _is_reshard(candidate: OpCandidate) -> bool:
    return isinstance(candidate.op, Reshard)


def _is_view(candidate: OpCandidate) -> bool:
    return isinstance(candidate.op, (Reshape, Transpose)) and candidate.duration_ns == 0


def _checked_add(total: int, value: int, context: str) -> int:
    result = total + value
    if result < 0 or result > _INT64_MAX:
        raise ValueError(f"P3: {context} exceeds OR-Tools integer domain")
    return result


def _checked_mul(left: int, right: int, context: str) -> int:
    if left < 0 or right < 0 or (left and right > _INT64_MAX // left):
        raise ValueError(f"P3: {context} exceeds OR-Tools integer domain")
    return left * right


def _region_chain(problem: PlanningProblem, region_id: int | None) -> tuple[RegionInfo, ...]:
    chain: list[RegionInfo] = []
    while region_id is not None:
        region = problem.regions[region_id]
        chain.append(region)
        region_id = region.parent_region_id
    return tuple(reversed(chain))


def _horizon(problem: PlanningProblem) -> int:
    horizon = 0
    for candidate_id, candidate in problem.candidates.items():
        if candidate.duration_ns <= 0:
            continue
        duration = candidate.duration_ns
        for region in _region_chain(problem, problem.candidate_enclosing_regions.get(candidate_id)):
            duration = _checked_mul(duration, region.trip_count, "horizon")
        horizon = _checked_add(horizon, duration, "horizon")
    return horizon


def _tensor_mesh_count(type: Type) -> int:
    if not isinstance(type, TensorType) or not isinstance(type.layout, ShardLayout):
        return 1
    shape = type.layout.mesh.layout.shape
    count = shape[0]
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"P3: bucket Mesh count must be a positive integer, got {count!r}")
    return count


def _mesh_offset(type: Type) -> int | None:
    if not isinstance(type, TensorType) or not isinstance(type.layout, ShardLayout):
        return None
    layout = type.layout.mesh.layout
    return layout.offset if isinstance(layout, ComposedLayout) else None


def _buckets_for_value(problem: PlanningProblem, value_id: int) -> tuple[int, ...]:
    return tuple(
        bucket_id for bucket_id, bucket in problem.buckets.items() if bucket.value_id == value_id
    )


def _buckets_by_type(problem: PlanningProblem, value_id: int) -> dict[int, int]:
    return {
        bucket.type_id: bucket_id
        for bucket_id, bucket in problem.buckets.items()
        if bucket.value_id == value_id
    }


def _source_value_ids(problem: PlanningProblem) -> tuple[int, ...]:
    return tuple(
        value_id
        for value_id, value in problem.values.items()
        if value.producer_site_id is None and value.function_path == ()
    )


def _result_region_ids(problem: PlanningProblem) -> dict[int, int]:
    result_regions: dict[int, int] = {}
    for region_id, region in problem.regions.items():
        for value_id in region.result_value_ids:
            result_regions[value_id] = region_id
    return result_regions


def _descendant_regions(problem: PlanningProblem, region_id: int) -> set[int]:
    descendants = {region_id}
    changed = True
    while changed:
        changed = False
        for candidate_id, region in problem.regions.items():
            if region.parent_region_id in descendants and candidate_id not in descendants:
                descendants.add(candidate_id)
                changed = True
    return descendants


def _add_exactly_one(model: cp_model.CpModel, literals: list[cp_model.IntVar], label: str) -> None:
    if not literals:
        raise ValueError(f"P3: no selectable {label}")
    model.AddExactlyOne(literals)


def _build_model(problem: PlanningProblem) -> _CpModelState:
    horizon = _horizon(problem)
    model = cp_model.CpModel()
    pick_candidates = {
        candidate_id: model.NewBoolVar(f"pick_candidate_{candidate_id}")
        for candidate_id in sorted(problem.candidates)
    }
    pick_buckets = {
        bucket_id: model.NewBoolVar(f"pick_bucket_{bucket_id}")
        for bucket_id in sorted(problem.buckets)
    }

    for site_id in problem.site_order:
        _add_exactly_one(
            model,
            [pick_candidates[candidate_id] for candidate_id in problem.authored_candidates[site_id]],
            f"authored candidates for site {site_id}",
        )
    for value_id in _source_value_ids(problem):
        _add_exactly_one(
            model,
            [pick_buckets[bucket_id] for bucket_id in _buckets_for_value(problem, value_id)
             if problem.buckets[bucket_id].is_source],
            f"source buckets for value {value_id}",
        )
    for requirement in problem.requirements:
        _add_exactly_one(
            model,
            [pick_buckets[bucket_id] for bucket_id in requirement.bucket_ids],
            f"requirement buckets for value {requirement.value_id}",
        )
    for value_id, value in problem.values.items():
        if value.is_final_output:
            _add_exactly_one(
                model,
                [pick_buckets[bucket_id] for bucket_id in _buckets_for_value(problem, value_id)],
                f"function result buckets for value {value_id}",
            )

    for bucket_id, bucket in problem.buckets.items():
        if bucket.is_source:
            continue
        producers = [pick_candidates[candidate_id] for candidate_id in bucket.candidate_ids]
        model.Add(sum(producers) == pick_buckets[bucket_id])
    for candidate_id, candidate in problem.candidates.items():
        present = pick_candidates[candidate_id]
        for bucket_id in (*candidate.input_bucket_ids, *candidate.output_bucket_ids):
            model.AddImplication(present, pick_buckets[bucket_id])

    for candidate_id, candidate in problem.candidates.items():
        if not _is_reshard(candidate):
            continue
        output_bucket = candidate.output_bucket_ids[0]
        demand_literals: list[cp_model.IntVar] = []
        if problem.buckets[output_bucket].value_id in problem.root_value_ids:
            demand_literals.append(pick_buckets[output_bucket])
        demand_literals.extend(
            pick_buckets[other_candidate.input_bucket_ids[input_index]]
            for other_candidate_id, other_candidate in problem.candidates.items()
            for input_index, input_bucket in enumerate(other_candidate.input_bucket_ids)
            if input_bucket == output_bucket and other_candidate_id != candidate_id
        )
        for requirement in problem.requirements:
            if output_bucket in requirement.bucket_ids:
                demand_literals.append(pick_buckets[output_bucket])
        if demand_literals:
            model.AddBoolOr([pick_candidates[candidate_id].Not(), *demand_literals])
        else:
            model.Add(pick_candidates[candidate_id] == 0)

    starts: dict[int, cp_model.IntVar] = {}
    ends: dict[int, cp_model.IntVar] = {}
    ready = {
        bucket_id: model.NewIntVar(0, horizon, f"ready_{bucket_id}")
        for bucket_id in sorted(problem.buckets)
    }
    positive_intervals: dict[int, cp_model.IntervalVar] = {}
    for candidate_id, candidate in problem.candidates.items():
        if candidate.duration_ns <= 0:
            continue
        duration = candidate.duration_ns
        for region in _region_chain(problem, problem.candidate_enclosing_regions.get(candidate_id)):
            duration = _checked_mul(duration, region.trip_count, "candidate duration")
        start = model.NewIntVar(0, horizon, f"start_{candidate_id}")
        end = model.NewIntVar(0, horizon, f"end_{candidate_id}")
        starts[candidate_id] = start
        ends[candidate_id] = end
        interval = model.NewOptionalIntervalVar(
            start, duration, end, pick_candidates[candidate_id], f"execution_{candidate_id}"
        )
        positive_intervals[candidate_id] = interval
        model.Add(start == horizon).OnlyEnforceIf(pick_candidates[candidate_id].Not())
        model.Add(end == 0).OnlyEnforceIf(pick_candidates[candidate_id].Not())
        for input_bucket in candidate.input_bucket_ids:
            model.Add(start >= ready[input_bucket]).OnlyEnforceIf(
                pick_candidates[candidate_id]
            )

    result_regions = _result_region_ids(problem)
    for bucket_id, bucket in problem.buckets.items():
        if bucket.is_source:
            model.Add(ready[bucket_id] == 0)
    for candidate_id, candidate in problem.candidates.items():
        present = pick_candidates[candidate_id]
        if candidate.duration_ns == 0:
            for output_bucket in candidate.output_bucket_ids:
                if candidate.input_bucket_ids:
                    model.Add(ready[output_bucket] == ready[candidate.input_bucket_ids[0]]).OnlyEnforceIf(
                        present
                    )
            continue
        for output_bucket in candidate.output_bucket_ids:
            model.Add(ready[output_bucket] == ends[candidate_id]).OnlyEnforceIf(present)

    offsets: dict[int, cp_model.IntVar] = {}
    for bucket_id, bucket in problem.buckets.items():
        bucket_type = problem.types[bucket.type_id]
        count = _tensor_mesh_count(bucket_type)
        if count > problem.topology.size:
            raise ValueError(f"P3: bucket {bucket_id} count {count} exceeds root topology")
        offset = model.NewIntVar(0, problem.topology.size - count, f"offset_{bucket_id}")
        offsets[bucket_id] = offset
        fixed_offset = bucket.fixed_offset
        if fixed_offset is None:
            fixed_offset = _mesh_offset(bucket_type)
        if fixed_offset is not None:
            if not 0 <= fixed_offset <= problem.topology.size - count:
                raise ValueError(f"P3: bucket {bucket_id} fixed offset is outside topology")
            model.Add(offset == fixed_offset).OnlyEnforceIf(pick_buckets[bucket_id])

    region_starts: dict[int, cp_model.IntVar] = {
        region_id: model.NewIntVar(0, horizon, f"region_start_{region_id}")
        for region_id in problem.regions
    }
    region_ends: dict[int, cp_model.IntVar] = {
        region_id: model.NewIntVar(0, horizon, f"region_end_{region_id}")
        for region_id in problem.regions
    }
    region_members = {
        region_id: _descendant_regions(problem, region_id) for region_id in problem.regions
    }
    for region_id, region in problem.regions.items():
        members = region_members[region_id]
        member_candidates = [
            candidate_id
            for candidate_id, candidate_region in problem.candidate_enclosing_regions.items()
            if candidate_region in members and candidate_id in starts
        ]
        child_regions = [
            child_id for child_id, child in problem.regions.items()
            if child.parent_region_id == region_id
        ]
        starts_for_min = [starts[candidate_id] for candidate_id in member_candidates]
        starts_for_min.extend(
            region_starts[child_id] for child_id in child_regions if child_id in region_starts
        )
        ends_for_max = [ends[candidate_id] for candidate_id in member_candidates]
        ends_for_max.extend(
            region_ends[child_id] for child_id in child_regions if child_id in region_ends
        )
        if not starts_for_min or not ends_for_max:
            raise ValueError(f"P3: GridRegion {region_id} has no positive-duration work")
        model.AddMinEquality(region_starts[region_id], starts_for_min)
        model.AddMaxEquality(region_ends[region_id], ends_for_max)
        for carry in region.carry_infos:
            carry_values = (
                carry.init_value_id,
                carry.carried_value_id,
                carry.yield_value_id,
                carry.result_value_id,
            )
            for left_value, right_value in zip(carry_values, carry_values[1:]):
                left_by_type = _buckets_by_type(problem, left_value)
                right_by_type = _buckets_by_type(problem, right_value)
                for type_id in left_by_type.keys() & right_by_type.keys():
                    left_bucket = left_by_type[type_id]
                    right_bucket = right_by_type[type_id]
                    model.Add(pick_buckets[left_bucket] == pick_buckets[right_bucket])
                    model.Add(offsets[left_bucket] == offsets[right_bucket]).OnlyEnforceIf(
                        [pick_buckets[left_bucket], pick_buckets[right_bucket]]
                    )
        for carry in region.carry_infos:
            for bucket_id in _buckets_for_value(problem, carry.init_value_id):
                model.Add(ready[bucket_id] <= region_starts[region_id]).OnlyEnforceIf(
                    pick_buckets[bucket_id]
                )

    for candidate_id, candidate in problem.candidates.items():
        if candidate_id not in starts:
            continue
        candidate_region = problem.candidate_enclosing_regions.get(candidate_id)
        for bucket_id in candidate.input_bucket_ids:
            result_region = result_regions.get(problem.buckets[bucket_id].value_id)
            if result_region is None or candidate_region in _descendant_regions(problem, result_region):
                continue
            model.Add(starts[candidate_id] >= region_ends[result_region]).OnlyEnforceIf(
                pick_candidates[candidate_id]
            )

    makespan = model.NewIntVar(0, horizon, "makespan")
    makespan_terms = list(ends.values())
    makespan_terms.extend(region_ends.values())
    if makespan_terms:
        model.AddMaxEquality(makespan, makespan_terms)
    else:
        model.Add(makespan == 0)

    topology_intervals: list[cp_model.IntervalVar] = []
    time_intervals: list[cp_model.IntervalVar] = []
    for candidate_id, candidate in problem.candidates.items():
        if candidate_id not in starts or _is_reshard(candidate):
            continue
        output_offsets = [offsets[bucket_id] for bucket_id in candidate.output_bucket_ids]
        for output_offset in output_offsets[1:]:
            model.Add(output_offset == output_offsets[0]).OnlyEnforceIf(pick_candidates[candidate_id])
        if _is_view(candidate) and candidate.input_bucket_ids:
            for output_offset in output_offsets:
                model.Add(output_offset == offsets[candidate.input_bucket_ids[0]]).OnlyEnforceIf(
                    pick_candidates[candidate_id]
                )
        if candidate.input_bucket_ids:
            for dependency in (
                item for item in problem.dependencies
                if item.parent_candidate_id == candidate_id
            ):
                input_offset = offsets[dependency.child_bucket_id]
                output_offset = output_offsets[0]
                if dependency.placement_relation == "SAME_INTERVAL":
                    model.Add(input_offset == output_offset).OnlyEnforceIf(pick_candidates[candidate_id])
                elif dependency.placement_relation == "CONTAINED":
                    input_count = _tensor_mesh_count(
                        problem.types[problem.buckets[dependency.child_bucket_id].type_id]
                    )
                    model.Add(input_offset <= output_offset).OnlyEnforceIf(pick_candidates[candidate_id])
                    model.Add(
                        output_offset + candidate.topology_count <= input_offset + input_count
                    ).OnlyEnforceIf(pick_candidates[candidate_id])
        topology_interval = model.NewOptionalIntervalVar(
            output_offsets[0], candidate.topology_count,
            output_offsets[0] + candidate.topology_count,
            pick_candidates[candidate_id],
            f"topology_{candidate_id}",
        )
        topology_intervals.append(topology_interval)
        time_intervals.append(positive_intervals[candidate_id])
    if time_intervals:
        model.AddNoOverlap2D(time_intervals, topology_intervals)

    for region_id, region in problem.regions.items():
        parent_region = region.parent_region_id
        direct_candidates = [
            candidate_id for candidate_id, candidate_region in problem.candidate_enclosing_regions.items()
            if candidate_region == parent_region and candidate_id in starts
        ]
        for candidate_id in direct_candidates:
            before = model.NewBoolVar(f"candidate_{candidate_id}_before_region_{region_id}")
            present = pick_candidates[candidate_id]
            model.Add(starts[candidate_id] >= region_ends[region_id]).OnlyEnforceIf(
                [present, before.Not()]
            )
            model.Add(ends[candidate_id] <= region_starts[region_id]).OnlyEnforceIf(
                [present, before]
            )
        sibling_regions = [
            other_id for other_id, other in problem.regions.items()
            if other.parent_region_id == parent_region and other_id != region_id
        ]
        for other_id in sibling_regions:
            if other_id < region_id:
                continue
            before = model.NewBoolVar(f"region_{region_id}_before_{other_id}")
            model.Add(region_ends[region_id] <= region_starts[other_id]).OnlyEnforceIf(before)
            model.Add(region_ends[other_id] <= region_starts[region_id]).OnlyEnforceIf(before.Not())

    bandwidth_intervals: list[cp_model.IntervalVar] = []
    bandwidth_demands: list[int] = []
    device = problem.root.target.device
    for candidate_id, candidate in problem.candidates.items():
        if candidate_id not in starts or candidate.hbm_demand_bytes_per_ns <= 0:
            continue
        bandwidth_intervals.append(positive_intervals[candidate_id])
        demand = (
            math.ceil(device.hbm_bandwidth_bytes_per_second / 1_000_000_000)
            if _is_reshard(candidate)
            else candidate.hbm_demand_bytes_per_ns
        )
        bandwidth_demands.append(demand)
    if bandwidth_intervals:
        model.AddCumulative(
            bandwidth_intervals,
            bandwidth_demands,
            math.ceil(device.hbm_bandwidth_bytes_per_second / 1_000_000_000),
        )

    _add_capacity_resource(problem, model, pick_candidates, pick_buckets, starts, ends, makespan, horizon)
    return _CpModelState(
        model=model,
        pick_candidates=pick_candidates,
        pick_buckets=pick_buckets,
        starts=starts,
        ends=ends,
        ready=ready,
        offsets=offsets,
        makespan=makespan,
        horizon_ns=horizon,
    )


def _add_capacity_resource(
    problem: PlanningProblem,
    model: cp_model.CpModel,
    pick_candidates: dict[int, cp_model.IntVar],
    pick_buckets: dict[int, cp_model.IntVar],
    starts: dict[int, cp_model.IntVar],
    ends: dict[int, cp_model.IntVar],
    makespan: cp_model.IntVar,
    horizon: int,
) -> None:
    parent = list(range(len(problem.buckets)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for candidate in problem.candidates.values():
        if not _is_view(candidate) or not candidate.input_bucket_ids:
            continue
        for output_bucket in candidate.output_bucket_ids:
            union(output_bucket, candidate.input_bucket_ids[0])
    for region in problem.regions.values():
        for carry in region.carry_infos:
            carry_values = (
                carry.init_value_id,
                carry.carried_value_id,
                carry.yield_value_id,
                carry.result_value_id,
            )
            for left_value, right_value in zip(carry_values, carry_values[1:]):
                left_buckets = _buckets_by_type(problem, left_value)
                right_buckets = _buckets_by_type(problem, right_value)
                for type_id in left_buckets.keys() & right_buckets.keys():
                    union(left_buckets[type_id], right_buckets[type_id])

    groups: dict[int, list[int]] = {}
    for bucket_id in problem.buckets:
        groups.setdefault(find(bucket_id), []).append(bucket_id)

    intervals: list[cp_model.IntervalVar] = []
    demands: list[int] = []
    for group_id, bucket_ids in sorted(groups.items()):
        selected = [pick_buckets[bucket_id] for bucket_id in bucket_ids]
        active = model.NewBoolVar(f"allocation_active_{group_id}")
        for literal in selected:
            model.AddImplication(literal, active)
        model.AddBoolOr([active.Not(), *selected])
        start_terms: list[cp_model.IntVar] = []
        end_terms: list[cp_model.IntVar] = []
        for bucket_id in bucket_ids:
            value_id = problem.buckets[bucket_id].value_id
            if problem.values[value_id].producer_site_id is None:
                start_terms.append(_constant_or_selected(model, 0, pick_buckets[bucket_id], horizon,
                                                         f"source_start_{bucket_id}", default=horizon))
            if problem.values[value_id].is_const:
                start_terms.append(_constant_or_selected(model, 0, pick_buckets[bucket_id], horizon,
                                                         f"constant_start_{bucket_id}", default=horizon))
            for candidate_id, candidate in problem.candidates.items():
                if bucket_id in candidate.output_bucket_ids and candidate_id in starts:
                    start_terms.append(starts[candidate_id])
                if bucket_id in candidate.input_bucket_ids and candidate_id in ends:
                    end_terms.append(ends[candidate_id])
                if bucket_id in candidate.output_bucket_ids and candidate_id in ends:
                    end_terms.append(ends[candidate_id])
            if problem.values[value_id].is_final_output:
                final_end = model.NewIntVar(0, horizon, f"final_output_end_{bucket_id}")
                model.Add(final_end == makespan).OnlyEnforceIf(pick_buckets[bucket_id])
                model.Add(final_end == 0).OnlyEnforceIf(pick_buckets[bucket_id].Not())
                end_terms.append(final_end)
        if not start_terms:
            start_terms.append(model.NewConstant(0))
        if not end_terms:
            end_terms.append(model.NewConstant(0))
        minimum_start = model.NewIntVar(0, horizon, f"allocation_min_start_{group_id}")
        maximum_end = model.NewIntVar(0, horizon, f"allocation_max_end_{group_id}")
        model.AddMinEquality(minimum_start, start_terms)
        model.AddMaxEquality(maximum_end, end_terms)
        allocation_start = model.NewIntVar(0, horizon, f"allocation_start_{group_id}")
        allocation_end = model.NewIntVar(0, horizon, f"allocation_end_{group_id}")
        allocation_size = model.NewIntVar(0, horizon, f"allocation_size_{group_id}")
        model.Add(allocation_start == minimum_start).OnlyEnforceIf(active)
        model.Add(allocation_end == maximum_end).OnlyEnforceIf(active)
        model.Add(allocation_size == allocation_end - allocation_start)
        model.Add(allocation_start == 0).OnlyEnforceIf(active.Not())
        model.Add(allocation_end == 0).OnlyEnforceIf(active.Not())
        interval = model.NewOptionalIntervalVar(
            allocation_start, allocation_size, allocation_end, active,
            f"allocation_{group_id}",
        )
        intervals.append(interval)
        type_values = [problem.types[problem.buckets[bucket_id].type_id] for bucket_id in bucket_ids]
        byte_counts = [tensor_bytes(type) for type in type_values if isinstance(type, TensorType)]
        demands.append(max(byte_counts, default=0))
    if intervals:
        model.AddCumulative(intervals, demands, problem.root.target.device.hbm_capacity_bytes)


def _constant_or_selected(
    model: cp_model.CpModel,
    constant: int,
    present: cp_model.IntVar,
    horizon: int,
    name: str,
    *,
    default: int = 0,
) -> cp_model.IntVar:
    value = model.NewIntVar(0, horizon, name)
    model.Add(value == constant).OnlyEnforceIf(present)
    model.Add(value == default).OnlyEnforceIf(present.Not())
    return value


def _decode(
    problem: PlanningProblem,
    state: _CpModelState,
    solver: cp_model.CpSolver,
    status: int,
) -> PlanningSolution:
    selected_candidates = tuple(
        candidate_id for candidate_id in sorted(problem.candidates)
        if solver.Value(state.pick_candidates[candidate_id])
    )
    selected_buckets = tuple(
        bucket_id for bucket_id in sorted(problem.buckets)
        if solver.Value(state.pick_buckets[bucket_id])
    )
    intervals = tuple(
        (candidate_id, ExecutionInterval(solver.Value(state.starts[candidate_id]),
                                          solver.Value(state.ends[candidate_id])))
        for candidate_id in selected_candidates if candidate_id in state.starts
    )
    offsets = tuple(
        (bucket_id, solver.Value(state.offsets[bucket_id]))
        for bucket_id in selected_buckets
    )
    makespan = solver.Value(state.makespan)
    if status == cp_model.OPTIMAL:
        return PlanningSolution("OPTIMAL", selected_candidates, selected_buckets, intervals,
                                offsets, makespan, makespan, 0.0)
    best_bound = math.floor(solver.BestObjectiveBound())
    best_bound = max(0, min(best_bound, makespan))
    gap = (makespan - best_bound) / max(makespan, 1)
    return PlanningSolution("FEASIBLE_NOT_PROVEN", selected_candidates, selected_buckets,
                            intervals, offsets, makespan, best_bound, gap)


def _write_failure(options: ScheduleOptions, problem: PlanningProblem, error: Exception) -> None:
    if options.debug_dump_dir is None:
        return
    options.debug_dump_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "error": str(error),
        "root": problem.root.name,
        "status": type(error).__name__,
        "target": problem.root.target.name,
    }
    (options.debug_dump_dir / "solve_failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def solve_planning_problem(problem: PlanningProblem, options: ScheduleOptions) -> PlanningSolution:
    """Build and solve one private makespan CP-SAT model."""
    state: _CpModelState | None = None
    try:
        state = _build_model(problem)
        state.model.Minimize(state.makespan)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = options.timeout_seconds
        solver.parameters.num_search_workers = options.workers
        solver.parameters.random_seed = options.random_seed
        solver.parameters.cp_model_probing_level = 0
        status = solver.Solve(state.model)
        if status == cp_model.INFEASIBLE:
            raise ValueError(
                f"P3: infeasible CTA blueprint for root {problem.root.name!r} "
                f"on target {problem.root.target.name!r}"
            )
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError("P3: OR-Tools reported an invalid planning model")
        if status == cp_model.UNKNOWN:
            raise RuntimeError(
                f"P3: CP-SAT returned UNKNOWN without an incumbent for root {problem.root.name!r}"
            )
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"P3: unexpected CP-SAT status {solver.StatusName(status)}")
        solution = _decode(problem, state, solver, status)
        if options.debug_dump_dir is not None:
            write_debug_dumps(problem, solution, options.debug_dump_dir)
        return solution
    except (ValueError, RuntimeError) as error:
        _write_failure(options, problem, error)
        raise


__all__ = ["ExecutionInterval", "PlanningSolution", "solve_planning_problem"]
