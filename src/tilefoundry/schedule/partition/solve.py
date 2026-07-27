"""The CP-SAT solve over one closed partition problem.

Every number this model needs is already in the problem: durations, traffic
demands, capacities, and how many parallel positions there are. Nothing here
resolves a Target, projects a fact, or asks the hardware a question, so what the
solve minimises is fully determined by its input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

from ortools.sat.python import cp_model

from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.reshape import Reshape
from tilefoundry.ir.hir.tensor.transpose import Transpose
from tilefoundry.ir.types import TensorType, Type, tensor_bytes
from tilefoundry.ir.types.shard import ComposedLayout, ShardLayout
from tilefoundry.schedule import ScheduleOptions

from .problem import OpCandidate, PartitionProblem
from .program import RegionInfo

_INT64_MAX = (1 << 63) - 1


class PartitionSolveError(RuntimeError):
    """The closed problem has no schedule, or the solver could not decide."""


@dataclass(frozen=True)
class ExecutionInterval:
    """One selected candidate's half-open execution interval."""

    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class PartitionSolution:
    """What the solve selected, and how sure it is of the objective."""

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
    terminal_buckets: dict[int, cp_model.IntVar]
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
        raise PartitionSolveError(f"{context} exceeds the solver integer domain")
    return result


def _checked_mul(left: int, right: int, context: str) -> int:
    if left < 0 or right < 0 or (left and right > _INT64_MAX // left):
        raise PartitionSolveError(f"{context} exceeds the solver integer domain")
    return left * right


def _region_chain(
    problem: PartitionProblem, region_id: int | None
) -> tuple[RegionInfo, ...]:
    chain: list[RegionInfo] = []
    while region_id is not None:
        region = problem.regions[region_id]
        chain.append(region)
        region_id = region.parent_region_id
    return tuple(reversed(chain))


def _horizon(problem: PartitionProblem) -> int:
    horizon = 0
    for candidate_id, candidate in problem.candidates.items():
        if candidate.duration_ns <= 0:
            continue
        duration = candidate.duration_ns
        for region in _region_chain(
            problem, problem.candidate_enclosing_regions.get(candidate_id)
        ):
            duration = _checked_mul(duration, region.trip_count, "horizon")
        horizon = _checked_add(horizon, duration, "horizon")
    return horizon


def _tensor_mesh_count(type: Type) -> int:
    if not isinstance(type, TensorType) or not isinstance(type.layout, ShardLayout):
        return 1
    shape = type.layout.mesh.layout.shape
    count = shape[0]
    if not isinstance(count, int) or count <= 0:
        raise PartitionSolveError(
            f"bucket Mesh count must be a positive integer, got {count!r}"
        )
    return count


def _mesh_offset(type: Type) -> int | None:
    if not isinstance(type, TensorType) or not isinstance(type.layout, ShardLayout):
        return None
    layout = type.layout.mesh.layout
    return layout.offset if isinstance(layout, ComposedLayout) else None


def _buckets_for_value(problem: PartitionProblem, value_id: int) -> tuple[int, ...]:
    return tuple(
        bucket_id
        for bucket_id, bucket in problem.buckets.items()
        if bucket.value_id == value_id
    )


def _buckets_by_type(problem: PartitionProblem, value_id: int) -> dict[int, int]:
    return {
        bucket.type_id: bucket_id
        for bucket_id, bucket in problem.buckets.items()
        if bucket.value_id == value_id
    }


def _source_value_ids(problem: PartitionProblem) -> tuple[int, ...]:
    return tuple(
        value_id
        for value_id, value in problem.values.items()
        if value.role == "normal"
        and value.producer_site_id is None
        and value.function_path == ()
    )


def _result_region_ids(problem: PartitionProblem) -> dict[int, int]:
    result_regions: dict[int, int] = {}
    for region_id, region in problem.regions.items():
        for value_id in region.result_value_ids:
            result_regions[value_id] = region_id
    return result_regions


def _descendant_regions(problem: PartitionProblem, region_id: int) -> set[int]:
    descendants = {region_id}
    changed = True
    while changed:
        changed = False
        for candidate_id, region in problem.regions.items():
            if region.parent_region_id in descendants and candidate_id not in descendants:
                descendants.add(candidate_id)
                changed = True
    return descendants


def _allocation_groups(
    problem: PartitionProblem,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Conservative physical groups covering all possible bucket selections.

    Carry facts are unconditional in-place aliases. View aliases are selected
    candidate facts, so they are intentionally kept as singleton groups in the
    pre-solve capacity model. This can overestimate resident bytes, but cannot
    merge an unselected view path and undercount them.
    """
    bucket_ids = tuple(sorted(problem.buckets))
    parent = {bucket_id: bucket_id for bucket_id in bucket_ids}

    def find(bucket_id: int) -> int:
        while parent[bucket_id] != bucket_id:
            parent[bucket_id] = parent[parent[bucket_id]]
            bucket_id = parent[bucket_id]
        return bucket_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for region in problem.regions.values():
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
                    union(left_by_type[type_id], right_by_type[type_id])

    groups: dict[int, list[int]] = {}
    for bucket_id in bucket_ids:
        groups.setdefault(find(bucket_id), []).append(bucket_id)
    return tuple(
        (group_id, tuple(sorted(group_bucket_ids)))
        for group_id, group_bucket_ids in sorted(groups.items())
    )


def _add_exactly_one(
    model: cp_model.CpModel, literals: list[cp_model.IntVar], label: str
) -> None:
    if not literals:
        raise PartitionSolveError(f"no selectable {label}")
    model.AddExactlyOne(literals)


def _build_model(problem: PartitionProblem) -> _CpModelState:
    horizon = _horizon(problem)
    extent = problem.extent
    bandwidth_per_ns = math.ceil(
        problem.facts.memory_bandwidth_bytes_per_second / 1_000_000_000
    )
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
            [
                pick_candidates[candidate_id]
                for candidate_id in problem.authored_candidates[site_id]
            ],
            f"authored candidates for site {site_id}",
        )
    for value_id in _source_value_ids(problem):
        _add_exactly_one(
            model,
            [
                pick_buckets[bucket_id]
                for bucket_id in _buckets_for_value(problem, value_id)
                if problem.buckets[bucket_id].is_source
            ],
            f"source buckets for value {value_id}",
        )
    for requirement in problem.requirements:
        _add_exactly_one(
            model,
            [pick_buckets[bucket_id] for bucket_id in requirement.bucket_ids],
            f"requirement buckets for value {requirement.value_id}",
        )
    terminal_buckets: dict[int, cp_model.IntVar] = {}
    for value_id, value in problem.values.items():
        if value.is_final_output:
            value_bucket_ids = _buckets_for_value(problem, value_id)
            reshard_output_buckets = tuple(
                bucket_id
                for bucket_id in value_bucket_ids
                if any(
                    _is_reshard(problem.candidates[candidate_id])
                    for candidate_id in problem.buckets[bucket_id].candidate_ids
                )
            )
            if not reshard_output_buckets:
                _add_exactly_one(
                    model,
                    [pick_buckets[bucket_id] for bucket_id in value_bucket_ids],
                    f"function result buckets for value {value_id}",
                )
                continue
            terminal_literals = []
            for bucket_id in value_bucket_ids:
                terminal = model.NewBoolVar(f"terminal_root_bucket_{bucket_id}")
                terminal_buckets[bucket_id] = terminal
                model.AddImplication(terminal, pick_buckets[bucket_id])
                terminal_literals.append(terminal)
            _add_exactly_one(
                model, terminal_literals, f"function result buckets for value {value_id}"
            )

    for bucket_id, bucket in problem.buckets.items():
        if bucket.is_source or problem.values[bucket.value_id].role != "normal":
            continue
        producers = [
            pick_candidates[candidate_id] for candidate_id in bucket.candidate_ids
        ]
        model.Add(sum(producers) == pick_buckets[bucket_id])
    for candidate_id, candidate in problem.candidates.items():
        present = pick_candidates[candidate_id]
        for bucket_id in (*candidate.input_bucket_ids, *candidate.output_bucket_ids):
            model.AddImplication(present, pick_buckets[bucket_id])

    demand_literals_by_bucket: dict[int, list[cp_model.IntVar]] = {}
    for output_bucket, terminal in terminal_buckets.items():
        demand_literals_by_bucket.setdefault(output_bucket, []).append(terminal)
    for other_candidate_id, other_candidate in problem.candidates.items():
        for input_bucket in other_candidate.input_bucket_ids:
            demand_literals_by_bucket.setdefault(input_bucket, []).append(
                pick_candidates[other_candidate_id]
            )
    for requirement in problem.requirements:
        for bucket_id in requirement.bucket_ids:
            demand_literals_by_bucket.setdefault(bucket_id, []).append(
                pick_buckets[bucket_id]
            )

    for candidate_id, candidate in problem.candidates.items():
        if not _is_reshard(candidate) or candidate.site_id is not None:
            continue
        output_bucket = candidate.output_bucket_ids[0]
        demand_literals = tuple(
            dict.fromkeys(demand_literals_by_bucket.get(output_bucket, ()))
        )
        if demand_literals:
            demand = model.NewBoolVar(f"reshard_demand_{output_bucket}")
            for literal in demand_literals:
                model.AddImplication(literal, demand)
            model.AddBoolOr([demand.Not(), *demand_literals])
            model.AddImplication(pick_candidates[candidate_id], demand)
        else:
            model.Add(pick_candidates[candidate_id] == 0)

    starts: dict[int, cp_model.IntVar] = {}
    ends: dict[int, cp_model.IntVar] = {}
    ready = {
        bucket_id: model.NewIntVar(0, horizon, f"ready_{bucket_id}")
        for bucket_id in sorted(problem.buckets)
    }
    merged_geometry_sites = {
        site_id
        for site_id in problem.site_order
        if all(
            problem.candidates[candidate_id].duration_ns > 0
            and not _is_reshard(problem.candidates[candidate_id])
            for candidate_id in problem.authored_candidates[site_id]
        )
    }
    merged_geometry_candidates = {
        candidate_id
        for site_id in merged_geometry_sites
        for candidate_id in problem.authored_candidates[site_id]
    }
    positive_intervals: dict[int, cp_model.IntervalVar] = {}
    for candidate_id, candidate in problem.candidates.items():
        if candidate.duration_ns <= 0:
            continue
        duration = candidate.duration_ns
        for region in _region_chain(
            problem, problem.candidate_enclosing_regions.get(candidate_id)
        ):
            duration = _checked_mul(duration, region.trip_count, "candidate duration")
        start = model.NewIntVar(0, horizon, f"start_{candidate_id}")
        end = model.NewIntVar(0, horizon, f"end_{candidate_id}")
        starts[candidate_id] = start
        ends[candidate_id] = end
        if candidate_id not in merged_geometry_candidates:
            positive_intervals[candidate_id] = model.NewOptionalIntervalVar(
                start,
                duration,
                end,
                pick_candidates[candidate_id],
                f"execution_{candidate_id}",
            )
        else:
            model.Add(end == start + duration).OnlyEnforceIf(
                pick_candidates[candidate_id]
            )
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
                    model.Add(
                        ready[output_bucket] == ready[candidate.input_bucket_ids[0]]
                    ).OnlyEnforceIf(present)
            continue
        for output_bucket in candidate.output_bucket_ids:
            model.Add(ready[output_bucket] == ends[candidate_id]).OnlyEnforceIf(present)

    offsets: dict[int, cp_model.IntVar] = {}
    for bucket_id, bucket in problem.buckets.items():
        bucket_type = problem.types[bucket.type_id]
        count = _tensor_mesh_count(bucket_type)
        if count > extent:
            raise PartitionSolveError(
                f"bucket {bucket_id} count {count} exceeds topology extent {extent}"
            )
        offset = model.NewIntVar(0, extent - count, f"offset_{bucket_id}")
        offsets[bucket_id] = offset
        fixed_offset = bucket.fixed_offset
        if fixed_offset is None:
            fixed_offset = _mesh_offset(bucket_type)
        if fixed_offset is not None:
            if not 0 <= fixed_offset <= extent - count:
                raise PartitionSolveError(
                    f"bucket {bucket_id} fixed offset {fixed_offset} is outside "
                    f"topology extent {extent}"
                )
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
        region_id: _descendant_regions(problem, region_id)
        for region_id in problem.regions
    }
    for region_id, region in problem.regions.items():
        members = region_members[region_id]
        member_candidates = [
            candidate_id
            for candidate_id, candidate_region in problem.candidate_enclosing_regions.items()
            if candidate_region in members and candidate_id in starts
        ]
        child_regions = [
            child_id
            for child_id, child in problem.regions.items()
            if child.parent_region_id == region_id
        ]
        starts_for_min = [starts[candidate_id] for candidate_id in member_candidates]
        starts_for_min.extend(
            region_starts[child_id]
            for child_id in child_regions
            if child_id in region_starts
        )
        ends_for_max = [ends[candidate_id] for candidate_id in member_candidates]
        ends_for_max.extend(
            region_ends[child_id] for child_id in child_regions if child_id in region_ends
        )
        if not starts_for_min or not ends_for_max:
            raise PartitionSolveError(
                f"GridRegion {region_id} has no positive-duration work"
            )
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
                    model.Add(
                        offsets[left_bucket] == offsets[right_bucket]
                    ).OnlyEnforceIf(
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
            if result_region is None or candidate_region in _descendant_regions(
                problem, result_region
            ):
                continue
            model.Add(
                starts[candidate_id] >= region_ends[result_region]
            ).OnlyEnforceIf(pick_candidates[candidate_id])

    makespan = model.NewIntVar(0, horizon, "makespan")
    makespan_terms = list(ends.values())
    makespan_terms.extend(region_ends.values())
    if makespan_terms:
        model.AddMaxEquality(makespan, makespan_terms)
    else:
        model.Add(makespan == 0)

    topology_intervals: list[cp_model.IntervalVar] = []
    time_intervals: list[cp_model.IntervalVar] = []
    for site_id in sorted(merged_geometry_sites):
        site_start = model.NewIntVar(0, horizon, f"site_start_{site_id}")
        site_end = model.NewIntVar(0, horizon, f"site_end_{site_id}")
        site_duration = model.NewIntVar(0, horizon, f"site_duration_{site_id}")
        site_offset = model.NewIntVar(0, extent, f"site_offset_{site_id}")
        site_count = model.NewIntVar(1, extent, f"site_count_{site_id}")
        site_offset_end = model.NewIntVar(0, extent, f"site_offset_end_{site_id}")
        model.Add(site_offset_end == site_offset + site_count)
        for candidate_id in problem.authored_candidates[site_id]:
            candidate = problem.candidates[candidate_id]
            present = pick_candidates[candidate_id]
            model.Add(site_start == starts[candidate_id]).OnlyEnforceIf(present)
            model.Add(site_end == ends[candidate_id]).OnlyEnforceIf(present)
            model.Add(
                site_duration == ends[candidate_id] - starts[candidate_id]
            ).OnlyEnforceIf(present)
            model.Add(
                site_offset == offsets[candidate.output_bucket_ids[0]]
            ).OnlyEnforceIf(present)
            model.Add(site_count == candidate.topology_count).OnlyEnforceIf(present)
            output_offsets = [
                offsets[bucket_id] for bucket_id in candidate.output_bucket_ids
            ]
            for output_offset in output_offsets[1:]:
                model.Add(output_offset == output_offsets[0]).OnlyEnforceIf(present)
            for dependency in (
                item
                for item in problem.dependencies
                if item.parent_candidate_id == candidate_id
            ):
                input_offset = offsets[dependency.child_bucket_id]
                output_offset = output_offsets[0]
                if dependency.placement_relation == "SAME_INTERVAL":
                    model.Add(input_offset == output_offset).OnlyEnforceIf(present)
                elif dependency.placement_relation == "CONTAINED":
                    input_count = _tensor_mesh_count(
                        problem.types[problem.buckets[dependency.child_bucket_id].type_id]
                    )
                    model.Add(input_offset <= output_offset).OnlyEnforceIf(present)
                    model.Add(
                        output_offset + candidate.topology_count
                        <= input_offset + input_count
                    ).OnlyEnforceIf(present)
        time_intervals.append(
            model.NewIntervalVar(
                site_start, site_duration, site_end, f"site_execution_{site_id}"
            )
        )
        topology_intervals.append(
            model.NewIntervalVar(
                site_offset, site_count, site_offset_end, f"site_topology_{site_id}"
            )
        )
    for candidate_id, candidate in problem.candidates.items():
        if (
            candidate_id not in starts
            or _is_reshard(candidate)
            or candidate_id in merged_geometry_candidates
        ):
            continue
        output_offsets = [offsets[bucket_id] for bucket_id in candidate.output_bucket_ids]
        for output_offset in output_offsets[1:]:
            model.Add(output_offset == output_offsets[0]).OnlyEnforceIf(
                pick_candidates[candidate_id]
            )
        if _is_view(candidate) and candidate.input_bucket_ids:
            for output_offset in output_offsets:
                model.Add(
                    output_offset == offsets[candidate.input_bucket_ids[0]]
                ).OnlyEnforceIf(pick_candidates[candidate_id])
        if candidate.input_bucket_ids:
            for dependency in (
                item
                for item in problem.dependencies
                if item.parent_candidate_id == candidate_id
            ):
                input_offset = offsets[dependency.child_bucket_id]
                output_offset = output_offsets[0]
                if dependency.placement_relation == "SAME_INTERVAL":
                    model.Add(input_offset == output_offset).OnlyEnforceIf(
                        pick_candidates[candidate_id]
                    )
                elif dependency.placement_relation == "CONTAINED":
                    input_count = _tensor_mesh_count(
                        problem.types[problem.buckets[dependency.child_bucket_id].type_id]
                    )
                    model.Add(input_offset <= output_offset).OnlyEnforceIf(
                        pick_candidates[candidate_id]
                    )
                    model.Add(
                        output_offset + candidate.topology_count
                        <= input_offset + input_count
                    ).OnlyEnforceIf(pick_candidates[candidate_id])
        topology_intervals.append(
            model.NewOptionalIntervalVar(
                output_offsets[0],
                candidate.topology_count,
                output_offsets[0] + candidate.topology_count,
                pick_candidates[candidate_id],
                f"topology_{candidate_id}",
            )
        )
        time_intervals.append(positive_intervals[candidate_id])
    if time_intervals:
        model.AddNoOverlap2D(time_intervals, topology_intervals)

    for region_id, region in problem.regions.items():
        parent_region = region.parent_region_id
        direct_candidates = [
            candidate_id
            for candidate_id, candidate_region in problem.candidate_enclosing_regions.items()
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
            other_id
            for other_id, other in problem.regions.items()
            if other.parent_region_id == parent_region and other_id != region_id
        ]
        for other_id in sibling_regions:
            if other_id < region_id:
                continue
            before = model.NewBoolVar(f"region_{region_id}_before_{other_id}")
            model.Add(region_ends[region_id] <= region_starts[other_id]).OnlyEnforceIf(
                before
            )
            model.Add(region_ends[other_id] <= region_starts[region_id]).OnlyEnforceIf(
                before.Not()
            )

    bandwidth_intervals: list[cp_model.IntervalVar] = []
    bandwidth_demands: list[int] = []

    def add_bandwidth_group(candidate_ids: list[int], demand: int, label: str) -> None:
        literals = [pick_candidates[candidate_id] for candidate_id in candidate_ids]
        if len(literals) == 1:
            active = literals[0]
        else:
            active = model.NewBoolVar(f"bandwidth_active_{label}")
            for literal in literals:
                model.AddImplication(literal, active)
            model.AddBoolOr([active.Not(), *literals])
        start = model.NewIntVar(0, horizon, f"bandwidth_start_{label}")
        end = model.NewIntVar(0, horizon, f"bandwidth_end_{label}")
        duration = model.NewIntVar(0, horizon, f"bandwidth_duration_{label}")
        for candidate_id in candidate_ids:
            model.Add(start == starts[candidate_id]).OnlyEnforceIf(
                pick_candidates[candidate_id]
            )
            model.Add(end == ends[candidate_id]).OnlyEnforceIf(
                pick_candidates[candidate_id]
            )
            model.Add(
                duration == ends[candidate_id] - starts[candidate_id]
            ).OnlyEnforceIf(pick_candidates[candidate_id])
        bandwidth_intervals.append(
            model.NewOptionalIntervalVar(
                start, duration, end, active, f"bandwidth_{label}"
            )
        )
        bandwidth_demands.append(demand)

    for site_id in problem.site_order:
        groups: dict[int, list[int]] = {}
        for candidate_id in problem.authored_candidates[site_id]:
            candidate = problem.candidates[candidate_id]
            if candidate_id not in starts or candidate.hbm_demand_bytes_per_ns <= 0:
                continue
            demand = (
                bandwidth_per_ns
                if _is_reshard(candidate)
                else candidate.hbm_demand_bytes_per_ns
            )
            groups.setdefault(demand, []).append(candidate_id)
        for demand, candidate_ids in sorted(groups.items()):
            add_bandwidth_group(
                candidate_ids, demand, f"site_{site_id}_demand_{demand}"
            )

    for candidate_id, candidate in problem.candidates.items():
        if candidate.site_id is not None or candidate_id not in starts:
            continue
        if candidate.hbm_demand_bytes_per_ns <= 0:
            continue
        add_bandwidth_group(
            [candidate_id], bandwidth_per_ns, f"candidate_{candidate_id}"
        )
    if bandwidth_intervals:
        model.AddCumulative(bandwidth_intervals, bandwidth_demands, bandwidth_per_ns)

    _add_capacity_resource(
        problem, model, pick_buckets, starts, ends, makespan, horizon
    )
    return _CpModelState(
        model=model,
        pick_candidates=pick_candidates,
        pick_buckets=pick_buckets,
        terminal_buckets=terminal_buckets,
        starts=starts,
        ends=ends,
        ready=ready,
        offsets=offsets,
        makespan=makespan,
        horizon_ns=horizon,
    )


def _add_capacity_resource(
    problem: PartitionProblem,
    model: cp_model.CpModel,
    pick_buckets: dict[int, cp_model.IntVar],
    starts: dict[int, cp_model.IntVar],
    ends: dict[int, cp_model.IntVar],
    makespan: cp_model.IntVar,
    horizon: int,
) -> None:
    """Charge each allocation group its widest resident type for its lifetime."""
    intervals: list[cp_model.IntervalVar] = []
    demands: list[int] = []
    for group_id, bucket_ids in _allocation_groups(problem):
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
                start_terms.append(
                    _constant_or_selected(
                        model,
                        0,
                        pick_buckets[bucket_id],
                        horizon,
                        f"source_start_{bucket_id}",
                        default=horizon,
                    )
                )
            if problem.values[value_id].is_const:
                start_terms.append(
                    _constant_or_selected(
                        model,
                        0,
                        pick_buckets[bucket_id],
                        horizon,
                        f"constant_start_{bucket_id}",
                        default=horizon,
                    )
                )
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
            if problem.values[value_id].is_const:
                constant_end = model.NewIntVar(0, horizon, f"constant_end_{bucket_id}")
                model.Add(constant_end == makespan).OnlyEnforceIf(
                    pick_buckets[bucket_id]
                )
                model.Add(constant_end == 0).OnlyEnforceIf(
                    pick_buckets[bucket_id].Not()
                )
                end_terms.append(constant_end)
        if not start_terms:
            start_terms.append(model.NewConstant(0))
        if not end_terms:
            end_terms.append(model.NewConstant(0))
        minimum_start = model.NewIntVar(0, horizon, f"allocation_min_start_{group_id}")
        maximum_end = model.NewIntVar(0, horizon, f"allocation_max_end_{group_id}")
        allocation_size = model.NewIntVar(0, horizon, f"allocation_size_{group_id}")
        model.AddMinEquality(minimum_start, start_terms)
        model.AddMaxEquality(maximum_end, end_terms)
        model.Add(allocation_size == maximum_end - minimum_start).OnlyEnforceIf(active)
        model.Add(allocation_size == 0).OnlyEnforceIf(active.Not())
        intervals.append(
            model.NewOptionalIntervalVar(
                minimum_start,
                allocation_size,
                maximum_end,
                active,
                f"allocation_{group_id}",
            )
        )
        type_values = [
            problem.types[problem.buckets[bucket_id].type_id] for bucket_id in bucket_ids
        ]
        byte_counts = [
            tensor_bytes(type) for type in type_values if isinstance(type, TensorType)
        ]
        demands.append(max(byte_counts, default=0))
    if intervals:
        model.AddCumulative(intervals, demands, problem.facts.memory_capacity_bytes)


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
    problem: PartitionProblem,
    state: _CpModelState,
    solver: cp_model.CpSolver,
    status: int,
) -> PartitionSolution:
    selected_candidates = tuple(
        candidate_id
        for candidate_id in sorted(problem.candidates)
        if solver.Value(state.pick_candidates[candidate_id])
    )
    selected_buckets = tuple(
        bucket_id
        for bucket_id in sorted(problem.buckets)
        if solver.Value(state.pick_buckets[bucket_id])
    )
    intervals = tuple(
        (
            candidate_id,
            ExecutionInterval(
                solver.Value(state.starts[candidate_id]),
                solver.Value(state.ends[candidate_id]),
            ),
        )
        for candidate_id in selected_candidates
        if candidate_id in state.starts
    )
    offsets = tuple(
        (bucket_id, solver.Value(state.offsets[bucket_id]))
        for bucket_id in selected_buckets
    )
    makespan = solver.Value(state.makespan)
    if status == cp_model.OPTIMAL:
        return PartitionSolution(
            "OPTIMAL",
            selected_candidates,
            selected_buckets,
            intervals,
            offsets,
            makespan,
            makespan,
            0.0,
        )
    best_bound = math.floor(solver.BestObjectiveBound())
    best_bound = max(0, min(best_bound, makespan))
    gap = (makespan - best_bound) / max(makespan, 1)
    return PartitionSolution(
        "FEASIBLE_NOT_PROVEN",
        selected_candidates,
        selected_buckets,
        intervals,
        offsets,
        makespan,
        best_bound,
        gap,
    )


def _write_failure(
    options: ScheduleOptions, problem: PartitionProblem, error: Exception
) -> None:
    if options.debug_dump_dir is None:
        return
    options.debug_dump_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "error": str(error),
        "root": problem.root.name,
        "status": type(error).__name__,
        "target": problem.facts.spec.device_id,
    }
    (options.debug_dump_dir / "solve_failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def solve_partition_problem(
    problem: PartitionProblem, options: ScheduleOptions
) -> PartitionSolution:
    """Build and solve one makespan model over an already-closed problem."""
    try:
        state = _build_model(problem)
        state.model.Minimize(state.makespan)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = options.timeout_seconds
        solver.parameters.num_search_workers = options.workers
        solver.parameters.random_seed = options.random_seed
        status = solver.Solve(state.model)
        if status == cp_model.INFEASIBLE:
            raise PartitionSolveError(
                f"no feasible partition for root {problem.root.name!r} at "
                f"topology {problem.topology.name!r} on {problem.facts.spec.device_id}"
            )
        if status == cp_model.MODEL_INVALID:
            raise PartitionSolveError("the solver reported an invalid partition model")
        if status == cp_model.UNKNOWN:
            raise PartitionSolveError(
                f"the solver returned no incumbent for root {problem.root.name!r} "
                "within its time limit"
            )
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise PartitionSolveError(
                f"unexpected solver status {solver.StatusName(status)}"
            )
        return _decode(problem, state, solver, status)
    except (ValueError, RuntimeError) as error:
        _write_failure(options, problem, error)
        raise


__all__ = [
    "ExecutionInterval",
    "PartitionSolution",
    "PartitionSolveError",
    "solve_partition_problem",
]
