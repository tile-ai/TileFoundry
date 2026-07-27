"""The AMX scheduler registered at the core level."""

from __future__ import annotations

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.schedule import ScheduleError, ScheduleOptions
from tilefoundry.schedule.pipeline import (
    PipelineFacts,
    PipelineSchedulePlan,
    build_pipeline_problem,
    build_pipeline_program,
    export_pipeline_plan,
    solve_pipeline_problem,
)
from tilefoundry.schedule.registry import register_schedule

from .target import AmxTarget

TOPOLOGY = "core"


def schedule_core(
    module: Module,
    function: Function,
    target: AmxTarget,
    topology: object,
    options: object | None = None,
) -> PipelineSchedulePlan:
    """Build, close, solve, and export the AMX core pipeline schedule."""
    if function is not module.entry_function():
        raise ScheduleError(
            f"{TOPOLOGY} schedule requires the module entry function, got "
            f"{function.name!r}"
        )
    if options is not None and not isinstance(options, ScheduleOptions):
        raise ScheduleError(
            f"{TOPOLOGY} schedule options must be ScheduleOptions, got "
            f"{type(options).__name__}"
        )
    program = build_pipeline_program(module, function)
    facts = target.as_facts(PipelineFacts, program.facts_query(TOPOLOGY))
    problem = build_pipeline_problem(program, facts, topology)
    solution = solve_pipeline_problem(problem)
    return export_pipeline_plan(program, solution, target)


register_schedule(AmxTarget, TOPOLOGY)(schedule_core)


__all__ = ["TOPOLOGY", "schedule_core"]
