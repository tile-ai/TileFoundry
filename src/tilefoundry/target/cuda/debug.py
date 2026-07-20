"""Private stable JSON projections for requested CTA planning diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.types import TensorType

from .cost import tensor_bytes
from .planner import PlanningProblem
from .projection import project_physical_fusion


def _type_label(type_value: object) -> str:
    return repr(type_value)


def _problem_payload(problem: PlanningProblem) -> dict:
    device = problem.root.target.device
    return {
        "root": problem.root.name,
        "module": problem.module.name,
        "target": problem.root.target.name,
        "topology": {"name": problem.topology.name, "size": problem.topology.size},
        "values": [
            {
                "id": value_id,
                "leaf_path": list(value.leaf_path),
                "is_const": value.is_const,
                "is_final_output": value.is_final_output,
                "producer_site_id": value.producer_site_id,
                "availability_region": problem.value_availability_regions.get(value_id),
                "type_labels": [
                    _type_label(problem.types[bucket.type_id])
                    for bucket in problem.buckets.values() if bucket.value_id == value_id
                ],
            }
            for value_id, value in sorted(problem.values.items())
        ],
        "buckets": [
            {
                "id": bucket_id,
                "value_id": bucket.value_id,
                "type_id": bucket.type_id,
                "candidate_ids": list(bucket.candidate_ids),
                "fixed_offset": bucket.fixed_offset,
                "is_source": bucket.is_source,
            }
            for bucket_id, bucket in sorted(problem.buckets.items())
        ],
        "candidates": [
            {
                "id": candidate_id,
                "op": type(candidate.op).__name__,
                "site_id": candidate.site_id,
                "input_bucket_ids": list(candidate.input_bucket_ids),
                "output_bucket_ids": list(candidate.output_bucket_ids),
                "duration_ns": candidate.duration_ns,
                "topology_count": candidate.topology_count,
                "hbm_bytes": candidate.total_hbm_bytes,
                "hbm_demand_bytes_per_ns": candidate.hbm_demand_bytes_per_ns,
                "moved_bytes": candidate.moved_bytes,
                "enclosing_region": problem.candidate_enclosing_regions.get(candidate_id),
            }
            for candidate_id, candidate in sorted(problem.candidates.items())
        ],
        "dependencies": [
            {
                "parent_candidate_id": dependency.parent_candidate_id,
                "input_index": dependency.input_index,
                "child_bucket_id": dependency.child_bucket_id,
                "placement_relation": dependency.placement_relation,
            }
            for dependency in sorted(
                problem.dependencies,
                key=lambda item: (item.parent_candidate_id, item.input_index, item.child_bucket_id),
            )
        ],
        "requirements": [
            {"value_id": requirement.value_id, "bucket_ids": list(requirement.bucket_ids)}
            for requirement in problem.requirements
        ],
        "regions": [
            {
                "id": region_id,
                "parent_region_id": region.parent_region_id,
                "trip_count": region.trip_count,
                "operation_site_ids": list(region.operation_site_ids),
                "carry_infos": [
                    {
                        "init_value_id": carry.init_value_id,
                        "carried_value_id": carry.carried_value_id,
                        "yield_value_id": carry.yield_value_id,
                        "result_value_id": carry.result_value_id,
                    }
                    for carry in region.carry_infos
                ],
            }
            for region_id, region in sorted(problem.regions.items())
        ],
        "resources": {
            "hbm_capacity_bytes": device.hbm_capacity_bytes,
            "hbm_bandwidth_bytes_per_second": device.hbm_bandwidth_bytes_per_second,
        },
    }


def _solution_payload(problem: PlanningProblem, solution: "PlanningSolution") -> dict:
    opportunities, cuts = project_physical_fusion(problem, solution)
    selected = set(solution.selected_candidate_ids)
    interval_map = dict(solution.candidate_intervals_ns)
    selected_buckets = set(solution.selected_bucket_ids)
    allocations = []
    for bucket_id in sorted(selected_buckets):
        bucket = problem.buckets[bucket_id]
        value = problem.values[bucket.value_id]
        producer_ids = [
            candidate_id for candidate_id in selected
            if bucket_id in problem.candidates[candidate_id].output_bucket_ids
        ]
        consumer_ends = [
            interval_map[candidate_id].end_ns
            for candidate_id in selected
            if bucket_id in problem.candidates[candidate_id].input_bucket_ids
            and candidate_id in interval_map
        ]
        producer_ends = [
            interval_map[candidate_id].end_ns for candidate_id in producer_ids
            if candidate_id in interval_map
        ]
        allocations.append({
            "bucket_id": bucket_id,
            "bytes": tensor_bytes(problem.types[bucket.type_id])
            if isinstance(problem.types[bucket.type_id], TensorType) else 0,
            "start_ns": 0 if problem.values[bucket.value_id].producer_site_id is None else min(
                (interval_map[candidate_id].start_ns for candidate_id in producer_ids
                 if candidate_id in interval_map), default=0
            ),
            "end_ns": solution.makespan_ns if value.is_final_output else max(
                producer_ends + consumer_ends, default=0
            ),
            "offset": dict(solution.bucket_offsets)[bucket_id],
            "is_const": value.is_const,
        })
    return {
        "program": {
            "module": problem.module.name,
            "root": problem.root.name,
            "function_instances": [
                {"path": list(path), "name": function.name}
                for path, function in problem.function_instances
            ],
        },
        "status": solution.status,
        "selected_candidate_ids": list(solution.selected_candidate_ids),
        "selected_bucket_ids": list(solution.selected_bucket_ids),
        "candidate_intervals_ns": [
            {"candidate_id": candidate_id, "start_ns": interval.start_ns, "end_ns": interval.end_ns}
            for candidate_id, interval in solution.candidate_intervals_ns
        ],
        "bucket_offsets": [
            {"bucket_id": bucket_id, "offset": offset}
            for bucket_id, offset in solution.bucket_offsets
        ],
        "allocations": allocations,
        "resources": {
            "candidate_timeline": [
                {
                    "candidate_id": candidate_id,
                    "hbm_demand_bytes_per_ns": problem.candidates[candidate_id].hbm_demand_bytes_per_ns,
                    "reshard": isinstance(problem.candidates[candidate_id].op, Reshard),
                    "start_ns": interval.start_ns,
                    "end_ns": interval.end_ns,
                }
                for candidate_id, interval in solution.candidate_intervals_ns
            ],
        },
        "makespan_ns": solution.makespan_ns,
        "best_bound_ns": solution.best_bound_ns,
        "gap": solution.gap,
        "physical_fusion_opportunities": [edge.__dict__ for edge in opportunities],
        "reshard_cuts": [cut.__dict__ for cut in cuts],
    }


def write_debug_dumps(problem: PlanningProblem, solution: "PlanningSolution", directory: Path) -> None:
    """Write the two stable projections requested by schedule options."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("planning_problem.json", _problem_payload(problem)),
        ("planning_solution.json", _solution_payload(problem, solution)),
    ):
        (directory / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = ["write_debug_dumps"]
