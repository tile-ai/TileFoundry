"""Focused P4 materialization and public CTA-service coverage."""

from __future__ import annotations

import runpy
from dataclasses import dataclass, replace

import pytest
import torch

from tests.models.deepseek_v4_flash.moe import (
    deepseek_v4_flash_module,
    deepseek_v4_flash_moe,
)
from tests.models.qwen3_5_30b_a3b.static_online import qwen_static_online
from tests.schedule.test_preflight import _planner_helper, _planner_root
from tilefoundry.evaluator import evaluate
from tilefoundry.inspection import as_script
from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.verify import verify_module
from tilefoundry.schedule import Schedule, ScheduleOptions
from tilefoundry.schedule.constraints import (
    ScheduleConstraint,
    ScheduleConstraintMetadata,
)
from tilefoundry.target.cuda.materialize import materialize_planning_solution
from tilefoundry.target.cuda.planner import build_planning_problem
from tilefoundry.target.cuda.solver import solve_planning_problem


@dataclass(frozen=True)
class _Marker(IRMetadata):
    value: str


def _small_module() -> Module:
    return Module("small", (_planner_helper, _planner_root), "_planner_root")


def _small_solution():
    module = _small_module()
    problem = build_planning_problem(module, _planner_root)
    solution = solve_planning_problem(
        problem, ScheduleOptions(timeout_seconds=10, workers=1)
    )
    return module, problem, solution


def test_materialization_clones_helper_paths_and_preserves_values() -> None:
    module, problem, solution = _small_solution()

    rebuilt = materialize_planning_solution(problem, solution)

    assert rebuilt is not module
    assert rebuilt.entry_function() is not _planner_root
    names = [function.name for function in rebuilt.functions]
    assert names.count("_planner_helper__cta_1") == 1
    assert names.count("_planner_helper__cta_2") == 1
    assert "_planner_helper" not in names

    x = torch.arange(8, dtype=torch.float32)
    torch.testing.assert_close(
        evaluate(_planner_root, x, device="cpu"),
        evaluate(rebuilt.entry_function(), x, device="cpu"),
    )
    verify_module(rebuilt.functions)


def test_invalid_solution_fails_without_mutating_input_module() -> None:
    module, problem, solution = _small_solution()
    original_functions = module.functions
    original_body = _planner_root.body
    invalid = replace(
        solution,
        selected_candidate_ids=(*solution.selected_candidate_ids, max(problem.candidates) + 1),
    )

    with pytest.raises(RuntimeError, match="unknown candidate"):
        materialize_planning_solution(problem, invalid)

    assert module.functions == original_functions
    assert _planner_root.body is original_body


def test_materialization_preserves_unrelated_metadata_and_consumes_constraints() -> None:
    marker = _Marker("keep")
    constraint = ScheduleConstraintMetadata(constraints=(ScheduleConstraint(),))
    body = replace(_planner_root.body, metadata=(marker, constraint))
    root = replace(_planner_root, body=body, metadata=(marker, constraint))
    module = Module("small", (_planner_helper, root), root.name)
    problem = build_planning_problem(module, root)
    solution = solve_planning_problem(
        problem, ScheduleOptions(timeout_seconds=10, workers=1)
    )

    rebuilt = materialize_planning_solution(problem, solution)

    rebuilt_root = rebuilt.entry_function()
    assert rebuilt_root.metadata == (marker,)
    assert rebuilt_root.body.metadata == (marker,)
    assert all(
        not isinstance(value, ScheduleConstraintMetadata)
        for value in (*rebuilt_root.metadata, *rebuilt_root.body.metadata)
    )


def test_cuda_cta_service_defaults_and_reconstructable_debug_dump(tmp_path) -> None:
    module = _small_module()
    target = _planner_root.target
    service = target.service(Schedule, "cta")
    assert service is target.service(Schedule, "cta")

    default_result = service.solve(module, _planner_root)
    assert default_result.report.stage == "cta"

    result = service.solve(
        module,
        _planner_root,
        ScheduleOptions(timeout_seconds=10, workers=1, debug_dump_dir=tmp_path),
    )

    assert result.report.stage == "cta"
    dump = tmp_path / "materialized_hir.py"
    source = dump.read_text()
    compile(source, str(dump), "exec")
    namespace = runpy.run_path(str(dump))
    dumped = namespace["small"]
    assert isinstance(dumped, Module)
    assert dumped.entry == result.module.entry
    verify_module(dumped.functions)


def test_as_script_accepts_hir_module() -> None:
    source = as_script(_small_module())
    assert '@module(entry="_planner_root")' in source
    assert "_planner_helper" in source


def test_real_deepseek_cta_service_materializes_verified_module() -> None:
    service = deepseek_v4_flash_moe.target.service(Schedule, "cta")

    result = service.solve(
        deepseek_v4_flash_module,
        deepseek_v4_flash_moe,
        ScheduleOptions(timeout_seconds=60, workers=4),
    )

    assert result.report.status in {"OPTIMAL", "FEASIBLE_NOT_PROVEN"}
    assert result.module.entry_function().name == deepseek_v4_flash_moe.name
    assert any("__cta_" in function.name for function in result.module.functions)
    verify_module(result.module.functions)


def test_real_static_qwen_cta_service_materializes_verified_module() -> None:
    module = Module("qwen", (qwen_static_online,), "qwen_static_online")
    service = qwen_static_online.target.service(Schedule, "cta")

    result = service.solve(
        module,
        qwen_static_online,
        ScheduleOptions(timeout_seconds=60, workers=4),
    )

    assert result.report.status in {"OPTIMAL", "FEASIBLE_NOT_PROVEN"}
    assert result.module.entry_function().name == qwen_static_online.name
    verify_module(result.module.functions)
