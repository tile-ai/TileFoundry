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


@func(target="cuda")
def bf16_gemm_rmsnorm(
    x: Tensor[(64, 128), "bf16"],
    w: Tensor[(128, 64), "bf16"],
    weight: Tensor[(64,), "f32"],
) -> Tensor[(64, 64), "bf16"]:
    h = matmul(x, w)
    return rms_norm(h, weight)


def _module():
    return replace(bf16_gemm_rmsnorm, topologies=(Topology("cta", 1),))


def test_pipeline_closes_target_facts_before_solving_and_exports_stable_values() -> None:
    module = _module()
    function = module.entry_function()
    program = build_pipeline_program(module, function)
    facts = module.resolve_target().as_facts(PipelineFacts, program.facts_query("cta"))
    problem = build_pipeline_problem(program, facts, Topology("cta", 1))

    assert tuple(item.id for item in problem.statements) == ("MM", "RN")
    assert all(item.candidates for item in problem.statements)
    assert not hasattr(problem, "target")

    result = schedule(module, function, topology="cta")
    plan = result.plan
    assert tuple(item.id for item in plan.statements) == ("MM", "RN")
    assert all(item.end > item.start >= 0 for item in plan.statements)
    assert plan.to_json() == plan.to_json()
    assert {hole.statement_id for hole in plan.holes} == {"MM", "RN"}
    assert all(isinstance(relation, str) for hole in plan.holes for relation in hole.relations)


def test_pipeline_rejects_missing_statement_facts_before_solving() -> None:
    module = _module()
    program = build_pipeline_program(module, module.entry_function())
    facts = module.resolve_target().as_facts(PipelineFacts, program.facts_query("cta"))
    incomplete = replace(facts, instructions=facts.instructions[:-1])

    with pytest.raises(PipelineProblemError, match="do not match"):
        build_pipeline_problem(program, incomplete, Topology("cta", 1))
