from __future__ import annotations

import json

from tests.models.deepseek_v4_flash.moe import deepseek_v4_flash_module, deepseek_v4_flash_moe
from tests.models.qwen3_5_30b_a3b.static_online import qwen_static_online
from tests.schedule.test_preflight import _planner_root
from tilefoundry.ir.core.module import Module
from tilefoundry.schedule import ScheduleOptions
from tilefoundry.target.cuda.allocation import _allocation_groups
from tilefoundry.target.cuda.planner import build_planning_problem
from tilefoundry.target.cuda.projection import project_physical_fusion
from tilefoundry.target.cuda.report import project_schedule_report
from tilefoundry.target.cuda.solver import solve_planning_problem


def _problem():
    return build_planning_problem(Module("m", (_planner_root,), "_planner_root"), _planner_root)


def test_small_fixture_decodes_one_makespan_blueprint(tmp_path) -> None:
    problem = _problem()
    solution = solve_planning_problem(
        problem,
        ScheduleOptions(timeout_seconds=10, workers=1, debug_dump_dir=tmp_path),
    )

    assert solution.status == "OPTIMAL"
    assert len(solution.selected_candidate_ids) == len(problem.site_order)
    assert solution.makespan_ns == 3
    assert solution.best_bound_ns == solution.makespan_ns
    assert solution.gap == 0.0
    assert all(interval.end_ns > interval.start_ns for _, interval in solution.candidate_intervals_ns)
    assert all(
        0 <= offset < problem.topology.size
        for _, offset in solution.bucket_offsets
    )

    report = project_schedule_report(problem, solution, stage="cta")
    assert set(json.loads(report.to_json())) == {
        "root",
        "target",
        "stage",
        "status",
        "objective_name",
        "unit",
        "selected",
        "best_bound",
        "gap",
    }
    assert not (tmp_path / "model.pb").exists()
    assert (tmp_path / "planning_problem.json").exists()
    assert (tmp_path / "planning_solution.json").exists()

    opportunities, cuts = project_physical_fusion(problem, solution)
    assert opportunities or cuts


def test_static_qwen_root_decodes_closed_4096_trip_region() -> None:
    problem = build_planning_problem(
        Module("qwen", (qwen_static_online,), "qwen_static_online"), qwen_static_online
    )
    solution = solve_planning_problem(problem, ScheduleOptions(timeout_seconds=60, workers=4))

    region = next(iter(problem.regions.values()))
    assert region.trip_count == 4096
    assert len(region.carry_infos) == 3
    assert solution.status in {"OPTIMAL", "FEASIBLE_NOT_PROVEN"}
    assert solution.makespan_ns >= 0
    assert solution.best_bound_ns <= solution.makespan_ns
    selected = set(solution.selected_bucket_ids)
    offsets = dict(solution.bucket_offsets)
    group_by_bucket = {
        bucket_id: group_id
        for group_id, bucket_ids in _allocation_groups(problem)
        for bucket_id in bucket_ids
    }
    for carry in region.carry_infos:
        values = (
            carry.init_value_id,
            carry.carried_value_id,
            carry.yield_value_id,
            carry.result_value_id,
        )
        buckets_by_type = [
            {
                problem.buckets[bucket_id].type_id: bucket_id
                for bucket_id in selected
                if problem.buckets[bucket_id].value_id == value_id
            }
            for value_id in values
        ]
        assert all(mapping.keys() == buckets_by_type[0].keys() for mapping in buckets_by_type[1:])
        for type_id in buckets_by_type[0]:
            bucket_ids = [mapping[type_id] for mapping in buckets_by_type]
            assert len({offsets[bucket_id] for bucket_id in bucket_ids}) == 1
            assert len({group_by_bucket[bucket_id] for bucket_id in bucket_ids}) == 1


def test_real_deepseek_root_decodes_constrained_blueprint() -> None:
    problem = build_planning_problem(deepseek_v4_flash_module, deepseek_v4_flash_moe)
    solution = solve_planning_problem(problem, ScheduleOptions(timeout_seconds=60, workers=8))

    assert solution.status in {"OPTIMAL", "FEASIBLE_NOT_PROVEN"}
    assert solution.selected_candidate_ids
    assert solution.selected_bucket_ids
    for requirement in problem.requirements:
        assert set(requirement.bucket_ids) & set(solution.selected_bucket_ids)
    assert solution.makespan_ns >= solution.best_bound_ns >= 0
