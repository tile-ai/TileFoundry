"""Materialization of a solved CTA plan, and the public call that produces it."""

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
from tilefoundry.ir.constraints import (
    ScheduleConstraint,
    ScheduleConstraintMetadata,
)
from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.verify import verify_module
from tilefoundry.schedule import ScheduleOptions, schedule
from tilefoundry.target.cuda.materialize import materialize_planning_solution
from tilefoundry.target.cuda.planner import build_planning_problem
from tilefoundry.target.cuda.solver import solve_planning_problem


@dataclass(frozen=True)
class _Marker(IRMetadata):
    value: str


def _small_module() -> Module:
    return Module(
        "small",
        (_planner_helper, _planner_root.entry_function()),
        "_planner_root",
        target=_planner_root.resolve_target(),
        topologies=_planner_root.effective_topologies(),
    )


def _small_solution():
    module = _small_module()
    problem = build_planning_problem(module, module.entry_function())
    solution = solve_planning_problem(
        problem, ScheduleOptions(timeout_seconds=10, workers=1)
    )
    return module, problem, solution


def test_materialization_clones_helper_paths_and_preserves_values() -> None:
    module, problem, solution = _small_solution()

    rebuilt = materialize_planning_solution(problem, solution)

    assert rebuilt is not module
    assert rebuilt.entry_function() is not _planner_root.entry_function()
    names = [function.name for function in rebuilt.functions]
    assert names.count("_planner_helper__cta_1") == 1
    assert names.count("_planner_helper__cta_2") == 1
    assert "_planner_helper" not in names

    x = torch.arange(8, dtype=torch.float32)
    torch.testing.assert_close(
        evaluate(_planner_root.entry_function(), x, device="cpu"),
        evaluate(rebuilt.entry_function(), x, device="cpu"),
    )
    verify_module(rebuilt.functions)


def test_invalid_solution_fails_without_mutating_input_module() -> None:
    module, problem, solution = _small_solution()
    original_functions = module.functions
    original_body = _planner_root.entry_function().body
    invalid = replace(
        solution,
        selected_candidate_ids=(*solution.selected_candidate_ids, max(problem.candidates) + 1),
    )

    with pytest.raises(RuntimeError, match="unknown candidate"):
        materialize_planning_solution(problem, invalid)

    assert module.functions == original_functions
    assert _planner_root.entry_function().body is original_body


def test_materialization_preserves_unrelated_metadata_and_consumes_constraints() -> None:
    marker = _Marker("keep")
    constraint = ScheduleConstraintMetadata(constraints=(ScheduleConstraint(),))
    body = replace(_planner_root.entry_function().body, metadata=(marker, constraint))
    root = replace(_planner_root.entry_function(), body=body, metadata=(marker, constraint))
    module = Module(
        "small",
        (_planner_helper, root),
        root.name,
        target=_planner_root.resolve_target(),
        topologies=_planner_root.effective_topologies(),
    )
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


def test_cta_schedule_defaults_and_reconstructable_debug_dump(tmp_path) -> None:
    """Solver controls default, and the dump is the plan's program as source.

    The dumped file is imported back and verified, which is what makes it a
    debug artifact rather than a rendering: it reconstructs the same Module the
    plan carries.
    """
    module = _small_module()

    default_result = schedule(module, module.entry_function(), topology="cta")
    assert default_result.plan.report.topology == "cta"

    result = schedule(
        module,
        module.entry_function(),
        topology="cta",
        options=ScheduleOptions(timeout_seconds=10, workers=1, debug_dump_dir=tmp_path),
    )

    assert result.plan.report.topology == "cta"
    dump = tmp_path / "materialized_hir.py"
    source = dump.read_text()
    compile(source, str(dump), "exec")
    namespace = runpy.run_path(str(dump))
    dumped = namespace["small"]
    assert isinstance(dumped, Module)
    assert dumped.entry == result.plan.materialized.entry
    verify_module(dumped.functions)


def test_real_deepseek_cta_schedule_materializes_verified_module() -> None:
    """A real model schedules at the CTA level and the plan carries the split.

    The input Module comes back untouched -- what was rewritten is the plan's own
    materialized program, which is where the per-CTA functions appear.
    """
    result = schedule(
        deepseek_v4_flash_module,
        deepseek_v4_flash_moe,
        topology="cta",
        options=ScheduleOptions(timeout_seconds=60, workers=4),
    )

    assert result.module is deepseek_v4_flash_module
    assert result.function is deepseek_v4_flash_moe
    assert result.plan.report.status in {"OPTIMAL", "FEASIBLE_NOT_PROVEN"}
    materialized = result.plan.materialized
    assert materialized.entry_function().name == deepseek_v4_flash_moe.name
    assert any("__cta_" in function.name for function in materialized.functions)
    verify_module(materialized.functions)


def test_real_static_qwen_cta_schedule_materializes_verified_module() -> None:
    """The same public call on a second real model, whose grid is static."""
    module = qwen_static_online
    result = schedule(
        module,
        module.entry_function(),
        topology="cta",
        options=ScheduleOptions(timeout_seconds=60, workers=4),
    )

    assert result.plan.report.status in {"OPTIMAL", "FEASIBLE_NOT_PROVEN"}
    materialized = result.plan.materialized
    assert materialized.entry_function().name == module.entry_function().name
    verify_module(materialized.functions)


def test_materialization_keeps_the_context_an_inheriting_child_was_scheduled_under() -> None:
    """The rebuilt Module leaves the owner chain, so it must carry the resolved
    context: copying the declared fields would hand back a child whose Target no
    longer resolves, while callers still schedule and verify against it."""
    declared = _small_module()
    child = Module("child", declared.functions, declared.entry)
    owner = Module(
        "owner", (), declared.entry, modules=(child,),
        target=declared.resolve_target(),
        topologies=declared.effective_topologies(),
    )
    inheriting = owner.modules[0]
    assert inheriting.target is None and inheriting.topologies is None

    problem = build_planning_problem(inheriting, inheriting.entry_function())
    solution = solve_planning_problem(
        problem, ScheduleOptions(timeout_seconds=10, workers=1)
    )

    rebuilt = materialize_planning_solution(problem, solution)

    assert rebuilt.resolve_target() == declared.resolve_target()
    assert rebuilt.effective_topologies() == declared.effective_topologies()
