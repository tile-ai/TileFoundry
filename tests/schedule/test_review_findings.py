"""Focused regressions for the finalized P3 resource and projection contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
from ortools.sat.python import cp_model

from tests.models.qwen3_5_30b_a3b.static_online import qwen_static_online
from tilefoundry.ir.core.module import Module
from tilefoundry.parser.hir_parser import parse_script
from tilefoundry.schedule import ScheduleOptions
from tilefoundry.target import CudaTarget
from tilefoundry.target.base import Device
from tilefoundry.target.cuda.allocation import _allocation_groups
from tilefoundry.target.cuda.device import H200SXM
from tilefoundry.target.cuda.planner import build_planning_problem
from tilefoundry.target.cuda.projection import project_physical_fusion
from tilefoundry.target.cuda.solver import PlanningSolution, _build_model, solve_planning_problem


def _qwen_problem():
    return build_planning_problem(
        Module("qwen", (qwen_static_online,), "qwen_static_online"), qwen_static_online
    )


def _reshard_case(problem):
    for candidate_id, candidate in problem.candidates.items():
        if type(candidate.op).__name__ != "Reshard" or candidate.site_id is not None:
            continue
        output_bucket = candidate.output_bucket_ids[0]
        if any(output_bucket in requirement.bucket_ids for requirement in problem.requirements):
            continue
        consumers = tuple(
            other_id
            for other_id, other in problem.candidates.items()
            if other_id != candidate_id and output_bucket in other.input_bucket_ids
        )
        if consumers:
            return candidate_id, candidate, consumers[0]
    raise AssertionError("Qwen fixture has no synthesized Reshard with a consumer")


def _root_only_problem(problem, value_id: int):
    values = {
        current_id: replace(value, is_final_output=current_id == value_id)
        for current_id, value in problem.values.items()
    }
    return replace(
        problem,
        values=MappingProxyType(values),
        root_value_ids=(value_id,),
    )


def test_reshard_demand_uses_consumer_selection_and_terminal_root_demand() -> None:
    problem = _qwen_problem()
    reshard_id, reshard, consumer_id = _reshard_case(problem)
    problem = _root_only_problem(problem, problem.buckets[reshard.output_bucket_ids[0]].value_id)

    state = _build_model(problem)
    state.model.Add(state.pick_candidates[reshard_id] == 1)
    state.model.Add(state.pick_candidates[consumer_id] == 0)
    state.model.Add(state.terminal_buckets[reshard.input_bucket_ids[0]] == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 20
    assert solver.Solve(state.model) == cp_model.INFEASIBLE

    state = _build_model(problem)
    state.model.Add(state.pick_candidates[reshard_id] == 1)
    state.model.Add(state.terminal_buckets[reshard.output_bucket_ids[0]] == 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 20
    assert solver.Solve(state.model) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def test_capacity_groups_do_not_union_unselected_views() -> None:
    problem = _qwen_problem()
    group_by_bucket = {
        bucket_id: group_id
        for group_id, bucket_ids in _allocation_groups(problem)
        for bucket_id in bucket_ids
    }
    view = next(
        candidate
        for candidate in problem.candidates.values()
        if type(candidate.op).__name__ in {"Reshape", "Transpose"}
        and candidate.duration_ns == 0
        and candidate.input_bucket_ids
    )
    assert all(
        group_by_bucket[output_bucket] != group_by_bucket[view.input_bucket_ids[0]]
        for output_bucket in view.output_bucket_ids
    )


class _CapacityDevice(Device):
    name = "test_capacity"
    sm_count = 132
    hbm_bandwidth_bytes_per_second = 4_800_000_000_000

    def __init__(self, capacity: int) -> None:
        self.hbm_capacity_bytes = capacity

    def peak_for(self, dtype):
        return H200SXM().peak_for(dtype)


def _constant_problem(capacity: int):
    root = parse_script(
        '''from tilefoundry import func
from tilefoundry.dsl import ConstTensor, Tensor, tf
from tilefoundry.target import CudaTarget
from tilefoundry.ir.types.shard import Topology

@func(target=CudaTarget(), topologies=(Topology("cta", 1),))
def root(x: Tensor[(1024,), "f32"], weight: ConstTensor[(1024,), "f32"]) -> Tensor[(1024,), "f32"]:
    y = tf.add(weight, weight)
    return tf.add(x, y)
'''
    )
    root = replace(root, target=CudaTarget(device=_CapacityDevice(capacity)))
    return build_planning_problem(Module("constant", (root,), "root"), root)


def test_constant_root_weight_is_resident_through_makespan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="infeasible CTA blueprint"):
        solve_planning_problem(_constant_problem(12_288), ScheduleOptions(timeout_seconds=10, workers=1))

    solution = solve_planning_problem(
        _constant_problem(16_384),
        ScheduleOptions(timeout_seconds=10, workers=1, debug_dump_dir=tmp_path),
    )
    payload = json.loads((tmp_path / "planning_solution.json").read_text())
    assert solution.status == "OPTIMAL"
    constant_allocations = [allocation for allocation in payload["allocations"] if allocation["is_const"]]
    assert constant_allocations
    assert all(allocation["end_ns"] == solution.makespan_ns for allocation in constant_allocations)
    assert all(
        bucket_id not in other_bucket_ids
        for index, allocation in enumerate(payload["allocations"])
        for bucket_id in allocation["bucket_ids"]
        for other in payload["allocations"][index + 1:]
        for other_bucket_ids in [other["bucket_ids"]]
    )


def test_reshard_edges_are_cuts_not_fusion_opportunities() -> None:
    problem = _qwen_problem()
    dependency = next(
        dependency
        for dependency in problem.dependencies
        if type(problem.candidates[dependency.parent_candidate_id].op).__name__ == "Reshard"
        and dependency.placement_relation is not None
        and any(
            dependency.child_bucket_id in candidate.output_bucket_ids
            and type(candidate.op).__name__ != "Reshard"
            for candidate in problem.candidates.values()
        )
    )
    reshard_id = dependency.parent_candidate_id
    producer_id = next(
        candidate_id
        for candidate_id, candidate in problem.candidates.items()
        if dependency.child_bucket_id in candidate.output_bucket_ids
        and type(candidate.op).__name__ != "Reshard"
    )
    selected_buckets = tuple(
        sorted(
            set(problem.candidates[reshard_id].input_bucket_ids)
            | set(problem.candidates[reshard_id].output_bucket_ids)
            | set(problem.candidates[producer_id].output_bucket_ids)
        )
    )
    solution = PlanningSolution(
        "OPTIMAL",
        (producer_id, reshard_id),
        selected_buckets,
        (),
        (),
        0,
        0,
        0.0,
    )
    opportunities, cuts = project_physical_fusion(problem, solution)
    assert any(cut.candidate_id == reshard_id for cut in cuts)
    assert all(edge.consumer_candidate_id != reshard_id for edge in opportunities)
