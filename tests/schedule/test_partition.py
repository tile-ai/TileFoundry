"""Closed spatial partition scheduling through the public boundary."""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import replace

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import matmul, rms_norm
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import PlanVerificationError, ScheduleError, ScheduleOptions, schedule
from tilefoundry.schedule.partition import (
    PartitionFacts,
    PartitionFactsError,
    PartitionProblemError,
    build_partition_problem,
    build_partition_program,
    solve_partition_problem,
)
from tilefoundry.schedule.partition import problem as problem_module
from tilefoundry.schedule.partition import solve as solve_module

_PARTITION_PACKAGE = pathlib.Path(problem_module.__file__).parent


@func(target="cuda")
def gemm_norm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "bf16"]:
    h = matmul(x, w)
    return rms_norm(h, weight)


def _module(extent: int = 4):
    return replace(gemm_norm, topologies=(Topology("cta", extent),))


def _closed(extent: int = 4):
    module = _module(extent)
    function = module.entry_function()
    program = build_partition_program(module, function)
    facts = module.resolve_target().as_facts(
        PartitionFacts, program.facts_query("cta")
    )
    return module, function, program, facts


def test_partition_schedules_through_the_public_operation_without_rewriting() -> None:
    module, function, _, _ = _closed()

    result = schedule(module, function, topology="cta")

    assert result.module is module
    assert result.function is function
    assert result.topology == Topology("cta", 4)
    assert result.plan.topology == "cta"
    assert result.plan.proof.objective_ns > 0
    assert result.plan.proof.best_bound_ns <= result.plan.proof.objective_ns
    assert result.plan.root_results


def test_partition_and_pipeline_answer_different_levels_of_one_module() -> None:
    module = replace(
        gemm_norm, topologies=(Topology("cta", 4), Topology("thread", 128))
    )
    function = module.entry_function()

    partition = schedule(module, function, topology="cta").plan
    pipeline = schedule(module, function, topology="thread").plan

    assert type(partition) is not type(pipeline)
    assert partition.topology == "cta"
    assert hasattr(partition, "placements")
    assert hasattr(pipeline, "statements")


def test_partition_program_states_the_program_without_asking_the_hardware() -> None:
    _, _, program, _ = _closed()

    assert program.sites
    assert program.root_value_ids
    assert all(
        base.storage.name.lower() == "gmem"
        for base in program.value_base_types.values()
    )
    assert not any(
        field.name in {"target", "facts", "device"}
        for field in dataclasses.fields(program)
    )


def test_partition_problem_closes_every_hardware_number_before_solving() -> None:
    _, _, program, facts = _closed()

    problem = build_partition_problem(program, facts, Topology("cta", 4))

    assert problem.facts is facts
    assert facts.parallel_units > 0
    assert facts.memory_bandwidth_bytes_per_second > 0
    assert facts.memory_capacity_bytes > 0
    assert facts.peak_flops_per_second
    assert all(
        not hasattr(candidate, "capacity_bytes")
        for candidate in problem.candidates.values()
    )
    assert all(candidate.duration_ns >= 0 for candidate in problem.candidates.values())


def test_partition_modules_never_reach_a_target() -> None:
    """The closed stages must hold numbers, not a machine to ask again."""
    forbidden = (
        "resolve_target",
        "as_facts",
        "CudaTarget",
        "AmxTarget",
        ".device.",
        ".architecture.",
        "TargetSpecRef.of",
    )
    for path in sorted(_PARTITION_PACKAGE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path.name} reaches a target through {name}"


def test_partition_prices_computation_and_reshard_as_one_candidate_concept() -> None:
    _, _, program, facts = _closed()

    problem = build_partition_problem(program, facts, Topology("cta", 4))

    authored = [
        candidate for candidate in problem.candidates.values() if candidate.site_id is not None
    ]
    synthesized = [
        candidate for candidate in problem.candidates.values() if candidate.site_id is None
    ]
    assert authored
    assert all(type(candidate).__name__ == "OpCandidate" for candidate in authored)
    assert all(type(candidate).__name__ == "OpCandidate" for candidate in synthesized)
    for candidate in synthesized:
        assert candidate.moved_bytes > 0
        assert candidate.topology_count == 0


def test_partition_adds_no_reshard_where_an_authored_candidate_already_produces() -> None:
    _, _, program, facts = _closed()

    problem = build_partition_problem(program, facts, Topology("cta", 4))

    for bucket in problem.buckets.values():
        producers = [problem.candidates[cid] for cid in bucket.candidate_ids]
        authored = [candidate for candidate in producers if candidate.site_id is not None]
        synthesized = [candidate for candidate in producers if candidate.site_id is None]
        assert not (authored and synthesized)


def test_partition_rejects_facts_projected_for_another_level() -> None:
    _, _, program, facts = _closed()

    with pytest.raises(PartitionProblemError, match="describe 'core'"):
        build_partition_problem(program, replace(facts, topology="core"), Topology("cta", 4))


def test_partition_rejects_an_extent_wider_than_the_hardware_states() -> None:
    _, _, program, facts = _closed()

    narrow = replace(facts, parallel_units=2)
    with pytest.raises(PartitionProblemError, match="exceeds the 2 parallel units"):
        build_partition_problem(program, narrow, Topology("cta", 4))


def test_partition_rejects_facts_missing_a_rate_it_must_charge_work_at() -> None:
    _, _, program, facts = _closed()

    with pytest.raises(PartitionFactsError, match="no dense peak rate"):
        build_partition_problem(
            program, replace(facts, peak_flops_per_second=()), Topology("cta", 4)
        )


def test_partition_facts_reject_a_level_the_target_does_not_partition() -> None:
    module, _, program, _ = _closed()

    with pytest.raises(ValueError, match="no partition facts for 'thread'"):
        module.resolve_target().as_facts(
            PartitionFacts, program.facts_query("thread")
        )


def test_partition_requires_a_level_the_program_declares() -> None:
    module = replace(gemm_norm, topologies=(Topology("thread", 128),))

    with pytest.raises(ScheduleError, match="cta"):
        schedule(module, module.entry_function(), topology="cta")


def test_partition_plan_verification_rejects_an_operation_on_an_unplaced_value() -> None:
    module, function, program, facts = _closed()
    problem = build_partition_problem(program, facts, Topology("cta", 4))
    solution = solve_partition_problem(problem, ScheduleOptions())
    plan = schedule(module, function, topology="cta").plan

    broken = replace(plan, placements=())
    with pytest.raises(PlanVerificationError, match="unplaced value bucket"):
        broken.verify(module, function, Topology("cta", 4))
    assert solution.status in ("OPTIMAL", "FEASIBLE_NOT_PROVEN")


def test_partition_plan_verification_rejects_a_bound_above_its_own_objective() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    broken = replace(
        plan, proof=replace(plan.proof, best_bound_ns=plan.proof.objective_ns + 1)
    )
    with pytest.raises(PlanVerificationError, match="bound above its own objective"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_renders_and_serializes_the_same_selection_every_time() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    assert plan.to_json() == plan.to_json()
    assert plan.render() == plan.render()
    assert "partition cta" in plan.render()


def test_partition_solve_reports_its_own_failures_rather_than_the_solver_status() -> None:
    assert issubclass(solve_module.PartitionSolveError, RuntimeError)
