"""Closed spatial partition scheduling through the public boundary."""

from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import replace

import pytest
from ortools.sat.python import cp_model

from tests.models.deepseek_v4_flash.moe import deepseek_v4_flash_module
from tests.models.qwen3_5_30b_a3b.static_online import qwen_static_online
from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import matmul, rms_norm
from tilefoundry.inspection.python_printer import as_script
from tilefoundry.ir.types import TensorType
from tilefoundry.ir.types.shard import ShardLayout, Topology
from tilefoundry.schedule import PlanVerificationError, ScheduleError, ScheduleOptions, schedule
from tilefoundry.schedule.partition import (
    PartitionedOperation,
    PartitionFacts,
    PartitionFactsError,
    PartitionProblemError,
    PartitionSchedulePlan,
    PlacedValue,
    PositionInterval,
    build_partition_problem,
    build_partition_program,
)
from tilefoundry.schedule.partition import problem as problem_module
from tilefoundry.schedule.partition import solve as solve_module

_SOLVER = ScheduleOptions(timeout_seconds=60, workers=8)

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
    assert hasattr(partition, "values")
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


def test_partition_plan_names_values_and_operations_from_the_authored_program() -> None:
    module, function, _, _ = _closed()

    plan = schedule(module, function, topology="cta").plan

    names = {value.id for value in plan.values}
    assert {"x", "w", "weight"} <= names
    for value in plan.values:
        assert isinstance(value.type, TensorType)
    producers = {value.producer_id for value in plan.values if value.producer_id}
    assert producers <= {operation.id for operation in plan.operations}
    assert plan.root_results
    assert set(plan.root_results) <= names


def test_partition_plan_states_a_reshard_as_an_operation_not_a_side_channel() -> None:
    """A moved value is one of the plan's own operations, with both placements."""
    module = qwen_static_online
    plan = schedule(
        module,
        module.entry_function(),
        topology="cta",
        options=_SOLVER,
    ).plan

    reshards = tuple(
        operation for operation in plan.operations if operation.operation == "Reshard"
    )
    assert reshards
    assert not hasattr(plan, "routes")
    assert not hasattr(plan, "report")
    values = {value.id: value for value in plan.values}
    for reshard in reshards:
        assert reshard.positions is None
    synthesized = tuple(item for item in reshards if item.synthesized)
    assert synthesized
    for reshard in synthesized:
        assert len(reshard.input_ids) == 1
        assert len(reshard.output_ids) == 1
        source = values[reshard.input_ids[0]]
        target = values[reshard.output_ids[0]]
        assert source.type != target.type
        assert (source.type.shape, source.type.dtype) == (
            target.type.shape,
            target.type.dtype,
        )


def test_partition_plan_holds_one_value_in_two_placements_at_once() -> None:
    """A Reshard connects two placements of one tensor, so both are named."""
    module = qwen_static_online
    plan = schedule(
        module, module.entry_function(), topology="cta", options=_SOLVER
    ).plan

    qualified = tuple(value for value in plan.values if "@" in value.id)
    assert qualified
    bases = {value.id.split("@", 1)[0] for value in qualified}
    for base in bases:
        placements = tuple(
            value for value in qualified if value.id.split("@", 1)[0] == base
        )
        assert len(placements) > 1
        assert len({value.type for value in placements}) == len(placements)


def test_partition_plan_verification_rejects_an_operation_on_an_unplaced_value() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    broken = replace(plan, values=())
    with pytest.raises(PlanVerificationError, match="unplaced value"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_a_dangling_producer() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    produced = next(value for value in plan.values if value.producer_id)
    broken = replace(
        plan,
        values=tuple(
            replace(value, producer_id="nobody") if value is produced else value
            for value in plan.values
        ),
    )
    with pytest.raises(PlanVerificationError, match="which the plan does not run"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_a_producer_that_produces_it_not() -> None:
    """A named producer must actually list the placement among its outputs."""
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    produced = next(value for value in plan.values if value.producer_id)
    other = next(
        operation
        for operation in plan.operations
        if produced.id not in operation.output_ids
    )

    broken = replace(
        plan,
        values=tuple(
            replace(value, producer_id=other.id) if value is produced else value
            for value in plan.values
        ),
    )
    with pytest.raises(PlanVerificationError, match="does not produce it"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_a_consumer_that_reads_it_not() -> None:
    """A named consumer must actually list the placement among its inputs."""
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    read = next(value for value in plan.values if value.consumer_ids)
    other = next(
        operation
        for operation in plan.operations
        if read.id not in operation.input_ids
    )

    broken = replace(
        plan,
        values=tuple(
            replace(value, consumer_ids=(*value.consumer_ids, other.id))
            if value is read
            else value
            for value in plan.values
        ),
    )
    with pytest.raises(PlanVerificationError, match="does not read it"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_an_output_the_value_disowns() -> None:
    """The operation side of an edge must agree with the placement side."""
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    produced = next(value for value in plan.values if value.producer_id)

    broken = replace(
        plan,
        values=tuple(
            replace(value, producer_id=None) if value is produced else value
            for value in plan.values
        ),
    )
    with pytest.raises(PlanVerificationError, match="which names producer None"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_an_input_the_value_disowns() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    read = next(value for value in plan.values if value.consumer_ids)

    broken = replace(
        plan,
        values=tuple(
            replace(value, consumer_ids=()) if value is read else value
            for value in plan.values
        ),
    )
    with pytest.raises(PlanVerificationError, match="does not name it as a consumer"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_will_not_reach_a_root_through_a_fabricated_edge() -> None:
    """A root must be reachable through edges both ends agree on."""
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    real = next(operation for operation in plan.operations if operation.output_ids)

    orphan = PlacedValue(
        id="orphan",
        type=plan.values[0].type,
        producer_id=real.id,
        consumer_ids=(),
        positions=plan.values[0].positions,
    )
    broken = replace(
        plan, values=(*plan.values, orphan), root_results=("orphan",)
    )
    with pytest.raises(PlanVerificationError, match="does not produce it"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_a_placement_outside_the_level() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    first = plan.values[0]
    broken = replace(
        plan,
        values=(
            replace(first, positions=PositionInterval(3, 9)),
            *plan.values[1:],
        ),
    )
    with pytest.raises(PlanVerificationError, match="outside the 4 positions"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_two_operations_sharing_a_position() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    placed = next(
        operation
        for operation in plan.operations
        if operation.positions is not None and operation.interval is not None
    )

    twin = replace(placed, id=placed.id + ".twin", input_ids=(), output_ids=())
    broken = replace(plan, operations=(*plan.operations, twin))
    with pytest.raises(PlanVerificationError, match="overlapping positions"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_a_reshard_that_moves_nothing() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    held = next(value for value in plan.values if value.producer_id is None)
    identity = PartitionedOperation(
        id="reshard:identity",
        operation="Reshard",
        synthesized=True,
        input_ids=(held.id,),
        output_ids=(held.id,),
        positions=None,
        interval=None,
    )

    broken = replace(
        plan,
        values=tuple(
            replace(
                value,
                producer_id=identity.id,
                consumer_ids=(*value.consumer_ids, identity.id),
            )
            if value is held
            else value
            for value in plan.values
        ),
        operations=(*plan.operations, identity),
    )
    with pytest.raises(PlanVerificationError, match="moves nothing"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_a_root_it_cannot_reach() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    broken = replace(plan, root_results=("nowhere",))
    with pytest.raises(PlanVerificationError, match="unplaced"):
        broken.verify(module, function, Topology("cta", 4))

    orphan = replace(
        plan.values[-1], id="orphan", producer_id="reshard:missing", consumer_ids=()
    )
    dangling = replace(
        plan, values=(*plan.values, orphan), root_results=("orphan",)
    )
    with pytest.raises(PlanVerificationError, match="does not run"):
        dangling.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_rejects_a_level_it_did_not_decide() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    with pytest.raises(PlanVerificationError, match="the level declares 8"):
        plan.verify(module, function, Topology("cta", 8))


def test_partition_plan_verification_rejects_a_bound_above_its_own_objective() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    broken = replace(
        plan, proof=replace(plan.proof, best_bound_ns=plan.proof.objective_ns + 1)
    )
    with pytest.raises(PlanVerificationError, match="bound above its own objective"):
        broken.verify(module, function, Topology("cta", 4))


def test_partition_plan_verification_needs_no_solver() -> None:
    """Verification is a structural check, so it must not reach the solver."""
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    calls: list[object] = []

    original = cp_model.CpSolver.Solve

    def refuse(self, model, *args, **kwargs):  # pragma: no cover - must not run
        calls.append(model)
        raise AssertionError("verification solved a model")

    cp_model.CpSolver.Solve = refuse
    try:
        plan.verify(module, function, Topology("cta", 4))
    finally:
        cp_model.CpSolver.Solve = original
    assert calls == []


def test_partition_plan_renders_the_same_decision_as_text_and_json() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan

    assert plan.to_json() == plan.to_json()
    assert plan.render() == plan.render()
    data = json.loads(plan.to_json())
    text = plan.render()

    assert data["topology"] == plan.topology == "cta"
    assert data["extent"] == plan.extent
    assert data["proof"]["status"] == plan.proof.status
    assert data["root_results"] == list(plan.root_results)
    assert {item["id"] for item in data["values"]} == {
        value.id for value in plan.values
    }
    assert {item["id"] for item in data["operations"]} == {
        operation.id for operation in plan.operations
    }
    for item in data["values"]:
        assert item["id"] in text
        assert item["type"]["dtype"] in text
    for item in data["operations"]:
        assert item["id"] in text
        if item["interval"] is not None:
            assert f"[{item['interval']['start_ns']}, " in text


def test_partition_plan_serializes_the_selected_type_it_placed_a_value_in() -> None:
    module, function, _, _ = _closed()
    plan = schedule(module, function, topology="cta").plan
    data = json.loads(plan.to_json())

    by_id = {item["id"]: item for item in data["values"]}
    for value in plan.values:
        stated = by_id[value.id]["type"]
        assert stated["dtype"] == value.type.dtype.name
        assert stated["storage"] == value.type.storage.name.lower()
        assert stated["shape"] == [str(dim) for dim in value.type.shape]
        if isinstance(value.type.layout, ShardLayout):
            assert stated["layout"]["topology"] == "cta"
        else:
            assert stated["layout"] is None


def test_partition_returns_the_program_it_was_asked_about_unchanged() -> None:
    module, function, _, _ = _closed()
    before = as_script(module)

    result = schedule(module, function, topology="cta")

    assert result.module is module
    assert result.function is function
    assert as_script(result.module) == before


def test_partition_never_materializes_its_decision_into_hir() -> None:
    """The plan states where work goes; nothing here rewrites the program."""
    for path in sorted(_PARTITION_PACKAGE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "materialize" not in text
    plan_fields = {field.name for field in dataclasses.fields(PartitionSchedulePlan)}
    assert "module" not in plan_fields
    assert "materialized" not in plan_fields


def test_partition_plans_a_real_moe_function() -> None:
    module = deepseek_v4_flash_module
    result = schedule(
        module, module.entry_function(), topology="cta", options=_SOLVER
    )
    plan = result.plan

    assert result.module is module
    assert plan.proof.status in ("OPTIMAL", "FEASIBLE_NOT_PROVEN")
    assert plan.proof.best_bound_ns <= plan.proof.objective_ns
    assert plan.operations and plan.values and plan.root_results
    assert plan.to_json() == plan.to_json()


def test_partition_solve_reports_its_own_failures_rather_than_the_solver_status() -> None:
    assert issubclass(solve_module.PartitionSolveError, RuntimeError)
