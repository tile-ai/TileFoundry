"""Closed spatial partition scheduling through the public boundary.

The solver itself is not asked to be optimal here; what is asked is that every
plan it returns holds together, that a plan which does not is rejected with the
reason it failed, and that the numbers it solved against were closed before it
ran. Those are the only checks standing between a wrong plan and the code
generated from one, so each distinct way a plan can be wrong keeps its own
message: which edge disagreed, which position was outside the level, which two
operations shared a position.

Whether a real model's entry function plans and verifies is the corpus Schedule
witness's subject. The program here is small so the mutations below can name
exactly one thing each.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace

import pytest
from ortools.sat.python import cp_model

from tests.fixtures.static_online import static_online_attend
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
    PlacedValue,
    PositionInterval,
    build_partition_problem,
    build_partition_program,
)
from tilefoundry.schedule.partition import solve as solve_module
from tilefoundry.schedule.pipeline.problem import PipelineProblemError
from tilefoundry.target import CudaTarget

#: What the assertions below read is how a plan states a move, which any plan that
#: verifies states the same way.
_SOLVER = ScheduleOptions(workers=1, stop_at_first_solution=True)


@func(target=CudaTarget("nvidia.h200_sxm"))
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
    facts = module.resolve_target().get_facts(
        PartitionFacts, program.facts_query("cta")
    )
    return module, function, program, facts


@pytest.fixture(scope="module")
def solved():
    """One solved plan, shared by every test that only mutates a copy of it.

    Verification is a structural check over an immutable plan, so the tests below
    build broken variants with `replace` rather than re-solving. Solving once is
    not an optimisation of the assertions -- it is the same plan under every
    mutation, which is what makes the failures comparable.
    """
    module, function, _, _ = _closed()
    return module, function, schedule(module, function, topology="cta").plan


def test_partition_schedules_through_the_public_operation_without_rewriting() -> None:
    """The plan is a decision about the program, so the program comes back as the
    same objects and prints identically -- nothing about scheduling rewrites it."""
    module, function, _, _ = _closed()
    before = as_script(module)

    result = schedule(module, function, topology="cta")

    assert result.module is module
    assert result.function is function
    assert result.topology == Topology("cta", 4)
    assert result.plan.topology == "cta"
    assert result.plan.proof.objective_ns > 0
    assert result.plan.proof.best_bound_ns <= result.plan.proof.objective_ns
    assert result.plan.root_results
    assert as_script(result.module) == before


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


def test_partition_prices_a_move_as_one_more_candidate_and_only_where_needed() -> None:
    """A reshard is not a side channel: it is priced as an ordinary candidate of
    the same kind, with bytes moved and no topology of its own. And it is
    synthesised only where nothing authored already produces the placement -- a
    bucket holding both an authored producer and a synthesised one would be
    charging for a move that has an original."""
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

    for bucket in problem.buckets.values():
        producers = [problem.candidates[cid] for cid in bucket.candidate_ids]
        assert not (
            [item for item in producers if item.site_id is not None]
            and [item for item in producers if item.site_id is None]
        )


def test_partition_refuses_a_level_the_facts_and_the_program_do_not_share() -> None:
    """Three ways of asking about the wrong level, each answered before a solve.

    Facts projected for another level describe another machine's parallelism; a
    level the target does not partition has no facts to project at all; and a
    level the program never declared has no extent to place work across. All
    three used to be servable by whatever numbers happened to be at hand.
    """
    module, _, program, facts = _closed()

    with pytest.raises(PartitionProblemError, match="describe 'core'"):
        build_partition_problem(program, replace(facts, topology="core"), Topology("cta", 4))

    with pytest.raises(ValueError, match="no partition facts for 'thread'"):
        module.resolve_target().get_facts(
            PartitionFacts, program.facts_query("thread")
        )

    thread_only = replace(gemm_norm, topologies=(Topology("thread", 128),))
    with pytest.raises(ScheduleError, match="cta"):
        schedule(thread_only, thread_only.entry_function(), topology="cta")


def test_partition_refuses_hardware_it_cannot_charge_the_work_against() -> None:
    """An extent wider than the machine states and a missing rate are both
    refusals, not defaults: a problem that guessed either would return a plan
    priced against a machine nobody has."""
    _, _, program, facts = _closed()

    with pytest.raises(PartitionProblemError, match="exceeds the 2 parallel units"):
        build_partition_problem(program, replace(facts, parallel_units=2), Topology("cta", 4))

    with pytest.raises(PartitionFactsError, match="no dense peak rate"):
        build_partition_problem(
            program, replace(facts, peak_flops_per_second=()), Topology("cta", 4)
        )


def test_partition_plan_names_values_and_operations_from_the_authored_program(
    solved,
) -> None:
    _, _, plan = solved

    names = {value.id for value in plan.values}
    assert {"x", "w", "weight"} <= names
    for value in plan.values:
        assert isinstance(value.type, TensorType)
    producers = {value.producer_id for value in plan.values if value.producer_id}
    assert producers <= {operation.id for operation in plan.operations}
    assert plan.root_results
    assert set(plan.root_results) <= names


def test_partition_plan_states_a_reshard_as_an_operation_with_both_placements() -> None:
    """A moved value is one of the plan's own operations, and both placements of
    it are named values: same shape and dtype, different type, and more than one
    placement sharing a base name. A plan that reported the move on the side
    would leave a reader unable to say where a value is."""
    module = static_online_attend
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

    qualified = tuple(value for value in plan.values if "@" in value.id)
    assert qualified
    for base in {value.id.split("@", 1)[0] for value in qualified}:
        placements = tuple(
            value for value in qualified if value.id.split("@", 1)[0] == base
        )
        assert len(placements) > 1


def test_verification_rejects_an_edge_the_two_ends_do_not_agree_on(solved) -> None:
    """Every producer/consumer edge is stated twice, and both statements have to
    say the same thing.

    A named producer that does not list the placement among its outputs, a named
    consumer that does not read it, and either side disowning an edge the other
    still names, are four separate corruptions with four separate messages -- and
    all four are invisible to a check that only walked one direction.
    """
    module, function, plan = solved
    level = Topology("cta", 4)

    produced = next(value for value in plan.values if value.producer_id)
    other_operation = next(
        operation
        for operation in plan.operations
        if produced.id not in operation.output_ids
    )
    with pytest.raises(PlanVerificationError, match="does not produce it"):
        _with_value(plan, produced, producer_id=other_operation.id).verify(
            module, function, level
        )

    read = next(value for value in plan.values if value.consumer_ids)
    not_a_reader = next(
        operation for operation in plan.operations if read.id not in operation.input_ids
    )
    with pytest.raises(PlanVerificationError, match="does not read it"):
        _with_value(
            plan, read, consumer_ids=(*read.consumer_ids, not_a_reader.id)
        ).verify(module, function, level)

    with pytest.raises(PlanVerificationError, match="which names producer None"):
        _with_value(plan, produced, producer_id=None).verify(module, function, level)

    with pytest.raises(PlanVerificationError, match="does not name it as a consumer"):
        _with_value(plan, read, consumer_ids=()).verify(module, function, level)


def _with_value(plan, target, **changes):
    """*plan* with one placement replaced -- the only difference from the plan the
    solver returned, so a rejection can only be about that one field."""
    return replace(
        plan,
        values=tuple(
            replace(value, **changes) if value is target else value
            for value in plan.values
        ),
    )


def test_verification_rejects_a_value_nothing_runs_and_a_root_nothing_reaches(
    solved,
) -> None:
    """A plan has to place every value it uses and reach every result it claims.

    An operation over a value the plan does not place, a producer the plan does
    not run, a root that is not a placement at all, and a root reachable only
    through an edge one end does not confirm: each is a plan that describes work
    nobody could carry out, and each says which.
    """
    module, function, plan = solved
    level = Topology("cta", 4)

    with pytest.raises(PlanVerificationError, match="unplaced value"):
        replace(plan, values=()).verify(module, function, level)

    produced = next(value for value in plan.values if value.producer_id)
    with pytest.raises(PlanVerificationError, match="which the plan does not run"):
        _with_value(plan, produced, producer_id="nobody").verify(module, function, level)

    with pytest.raises(PlanVerificationError, match="unplaced"):
        replace(plan, root_results=("nowhere",)).verify(module, function, level)

    real = next(operation for operation in plan.operations if operation.output_ids)
    orphan = PlacedValue(
        id="orphan",
        type=plan.values[0].type,
        producer_id=real.id,
        consumer_ids=(),
        positions=plan.values[0].positions,
    )
    with pytest.raises(PlanVerificationError, match="does not produce it"):
        replace(
            plan, values=(*plan.values, orphan), root_results=("orphan",)
        ).verify(module, function, level)


def test_verification_rejects_a_placement_the_level_does_not_contain(solved) -> None:
    """A plan is solved for one extent of one level, so it is only meaningful
    against that extent: a placement outside the positions the level declares, and
    a level of a different width than the one solved for, are both refused."""
    module, function, plan = solved

    first = plan.values[0]
    with pytest.raises(PlanVerificationError, match="outside the 4 positions"):
        _with_value(plan, first, positions=PositionInterval(3, 9)).verify(
            module, function, Topology("cta", 4)
        )

    with pytest.raises(PlanVerificationError, match="the level declares 8"):
        plan.verify(module, function, Topology("cta", 8))


def test_verification_rejects_two_operations_sharing_a_position(solved) -> None:
    """Two operations placed on one position at one time is the one thing a
    spatial partition exists to decide, so an overlap is not a preference."""
    module, function, plan = solved
    placed = next(
        operation
        for operation in plan.operations
        if operation.positions is not None and operation.interval is not None
    )

    twin = replace(placed, id=placed.id + ".twin", input_ids=(), output_ids=())
    broken = replace(plan, operations=(*plan.operations, twin))
    with pytest.raises(PlanVerificationError, match="overlapping positions"):
        broken.verify(module, function, Topology("cta", 4))


def test_verification_rejects_a_reshard_that_moves_nothing(solved) -> None:
    """A move whose source and destination are the same placement is a cost the
    plan charges for work it does not do."""
    module, function, plan = solved
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
        _with_value(
            plan,
            held,
            producer_id=identity.id,
            consumer_ids=(*held.consumer_ids, identity.id),
        ),
        operations=(*plan.operations, identity),
    )
    with pytest.raises(PlanVerificationError, match="moves nothing"):
        broken.verify(module, function, Topology("cta", 4))


def test_verification_rejects_a_bound_above_its_own_objective(solved) -> None:
    """The proof is part of the plan: a lower bound above the objective it bounds
    is an arithmetic impossibility, and reading one as optimality would report a
    plan as proven that is not."""
    module, function, plan = solved

    broken = replace(
        plan, proof=replace(plan.proof, best_bound_ns=plan.proof.objective_ns + 1)
    )
    with pytest.raises(PlanVerificationError, match="bound above its own objective"):
        broken.verify(module, function, Topology("cta", 4))


def test_verification_needs_no_solver(solved) -> None:
    """Verification is a structural check, so it must not reach the solver."""
    module, function, plan = solved
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


def test_partition_plan_renders_the_same_decision_as_text_and_json(solved) -> None:
    """One decision, two renderings, and the type each value was placed in stated
    exactly -- a serialization that dropped the selected type would describe a
    placement without saying what is placed there."""
    _, _, plan = solved

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


def test_a_problem_that_cannot_be_formed_is_a_schedule_failure() -> None:
    """The algorithms' own failures are reachable as `ScheduleError`.

    A caller asks this layer to schedule something and catches what the layer
    raises; a capability the layer cannot serve is recorded against the same
    type. While these sat outside it, a limit of an algorithm could only be
    stated as a bare `ValueError` -- which is also what a caller passing nonsense
    gets, so a recorded limit and a caller's mistake were indistinguishable.
    """
    for error in (PartitionProblemError, PipelineProblemError):
        assert issubclass(error, ScheduleError), error.__name__
        # Still a ValueError, so every existing caller keeps catching it.
        assert issubclass(error, ValueError), error.__name__

    # A solve that finds nothing is the solver's own outcome rather than a
    # malformed request, and stays a plain runtime failure.
    assert issubclass(solve_module.PartitionSolveError, RuntimeError)
