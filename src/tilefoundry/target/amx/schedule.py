"""The AMX scheduler registered at the core level."""

from __future__ import annotations

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.schedule import ScheduleError, ScheduleOptions
from tilefoundry.target.services import Scheduler

from .target import AmxTarget

TOPOLOGY = "core"


def schedule_core(
    module: Module,
    function: Function,
    target: AmxTarget,
    topology: object,
    options: object | None = None,
) -> object:
    """Build, close, solve, and export the AMX core pipeline schedule."""
    from tilefoundry.schedule.pipeline import (  # noqa: PLC0415
        PipelineFacts,
        build_pipeline_problem,
        build_pipeline_program,
        export_pipeline_plan,
        solve_pipeline_problem,
    )

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
    facts = target.get_facts(PipelineFacts, program.facts_query(TOPOLOGY))
    problem = build_pipeline_problem(program, facts, topology)
    solution = solve_pipeline_problem(problem)
    return export_pipeline_plan(program, solution, target)


def amx_scheduler(topology: str) -> Scheduler | None:
    """Construct the AMX scheduler requested for one topology level."""
    if topology == TOPOLOGY:
        return Scheduler(TOPOLOGY, schedule_core)
    return None


__all__ = ["TOPOLOGY", "amx_scheduler", "schedule_core"]
