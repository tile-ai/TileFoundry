"""Closed CUDA pipeline scheduling through the public boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tilefoundry import func
from tilefoundry.dsl import Tensor
from tilefoundry.dsl.tf import matmul, rms_norm
from tilefoundry.ir.types.shard import Topology
from tilefoundry.schedule import schedule
from tilefoundry.schedule.pipeline import (
    PipelineFacts,
    PipelineProblemError,
    build_pipeline_problem,
    build_pipeline_program,
)
from tilefoundry.schedule.pipeline.problem import (
    PipelineBufferProblem,
    PipelineProblem,
    PipelineStatementProblem,
)
from tilefoundry.schedule.pipeline.solve import (
    PipelineSolveError,
    solve_pipeline_problem,
)


@func(target="cuda")
def bf16_gemm_rmsnorm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "bf16"]:
    h = matmul(x, w)
    return rms_norm(h, weight)


def _module():
    return replace(bf16_gemm_rmsnorm, topologies=(Topology("cta", 1), Topology("thread", 128)))


def test_pipeline_closes_target_facts_before_solving_and_exports_stable_values() -> None:
    module = _module()
    function = module.entry_function()
    program = build_pipeline_program(module, function)
    facts = module.resolve_target().as_facts(PipelineFacts, program.facts_query("thread"))
    problem = build_pipeline_problem(program, facts, Topology("thread", 128))

    assert tuple(item.id for item in problem.statements) == ("MM", "RN")
    assert all(item.candidates for item in problem.statements)
    assert not hasattr(problem, "target")

    result = schedule(module, function, topology="thread")
    plan = result.plan
    assert tuple(item.id for item in plan.statements) == ("MM", "RN")
    assert all(item.end > item.start >= 0 for item in plan.statements)
    assert plan.to_json() == plan.to_json()
    assert {hole.statement_id for hole in plan.holes} == {"MM", "RN"}
    assert all(isinstance(relation, str) for hole in plan.holes for relation in hole.relations)


def _problem():
    module = _module()
    program = build_pipeline_program(module, module.entry_function())
    facts = module.resolve_target().as_facts(PipelineFacts, program.facts_query("thread"))
    return build_pipeline_problem(program, facts, Topology("thread", 128))


def test_a_buffer_that_carries_a_dependence_gets_more_than_one_slot() -> None:
    """The property that makes this a pipeline: `h` is the accumulator the
    matmul carries along k, so it has to hold two tiles at once."""
    problem = _problem()
    carried = {buffer.id: buffer.carried_distances for buffer in problem.buffers}
    assert carried["h"] == (("MM", (0, 0, 1)),)
    assert carried["x"] == ()

    ring = {buffer.id: buffer.ring_depth for buffer in solve_pipeline_problem(problem).buffers}
    assert ring["h"] == 2
    assert ring["x"] == 1
    assert set(ring.values()) != {1}


def test_ring_depth_counts_tiles_not_iterations() -> None:
    """A distance shorter than the tile it runs inside spans one tile."""
    def depth(distance: int, tile: int) -> int:
        problem = PipelineProblem(
            topology="thread",
            capacity_bytes=1024,
            statements=(
                PipelineStatementProblem(
                    id="S",
                    extents=(tile,),
                    candidates=(_candidate(),),
                    resources=(),
                    footprint_bytes=(),
                ),
            ),
            buffers=(
                PipelineBufferProblem(
                    id="b",
                    producer_ids=("S",),
                    consumer_ids=("S",),
                    carried_distances=(("S", (distance,)),),
                ),
            ),
        )
        return solve_pipeline_problem(problem).buffers[0].ring_depth

    assert depth(0, 8) == 1
    assert depth(1, 8) == 2
    assert depth(8, 8) == 2
    assert depth(9, 8) == 3
    assert depth(64, 8) == 9


def test_a_statement_records_what_it_holds_against_the_tile_store() -> None:
    """Capacity is recorded, not enforced -- but it has to be recorded."""
    problem = _problem()
    assert problem.capacity_bytes > 0
    held = dict(next(item for item in problem.statements if item.id == "MM").footprint_bytes)
    assert held == {"h": 8192, "w": 16384, "x": 16384}

    solution = solve_pipeline_problem(problem)
    ring = {buffer.id: buffer.ring_depth for buffer in solution.buffers}
    matmul_solution = next(item for item in solution.statements if item.id == "MM")
    assert matmul_solution.footprint_bytes == sum(
        held[name] * ring[name] for name in held
    )
    assert matmul_solution.fits_capacity is (
        matmul_solution.footprint_bytes <= problem.capacity_bytes
    )


def test_a_statement_too_wide_for_the_store_is_reported_not_dropped() -> None:
    problem = replace(_problem(), capacity_bytes=1)
    solution = solve_pipeline_problem(problem)

    assert len(solution.statements) == len(problem.statements)
    assert not any(item.fits_capacity for item in solution.statements)


def test_a_distance_measured_against_an_unknown_statement_is_refused() -> None:
    problem = _problem()
    broken = replace(
        problem,
        buffers=(
            replace(problem.buffers[0], carried_distances=(("nobody", (1,)),)),
            *problem.buffers[1:],
        ),
    )
    with pytest.raises(PipelineSolveError, match="unknown statement"):
        solve_pipeline_problem(broken)


def _candidate():
    """The one thing the solver reads off a candidate: how long it takes."""
    module = _module()
    program = build_pipeline_program(module, module.entry_function())
    facts = module.resolve_target().as_facts(PipelineFacts, program.facts_query("thread"))
    return facts.instructions[0].candidates[0]


def test_pipeline_rejects_missing_statement_facts_before_solving() -> None:
    module = _module()
    program = build_pipeline_program(module, module.entry_function())
    facts = module.resolve_target().as_facts(PipelineFacts, program.facts_query("thread"))
    incomplete = replace(facts, instructions=facts.instructions[:-1])

    with pytest.raises(PipelineProblemError, match="do not match"):
        build_pipeline_problem(program, incomplete, Topology("thread", 128))
