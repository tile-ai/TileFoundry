"""The CUDA scheduler registered at the CTA level.

This is the whole of what `schedule(..., topology="cta")` does on CUDA: build the
private planning problem, solve it, materialize the per-CTA program, and state
the objective. The steps below the boundary stay private to this package.
"""

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

from .target import CudaTarget

TOPOLOGY = "cta"


def _options(options: object | None) -> ScheduleOptions:
    """The solver controls to run under, defaulted rather than inferred."""
    if options is None:
        return ScheduleOptions()
    if not isinstance(options, ScheduleOptions):
        raise ScheduleError(
            f"{TOPOLOGY} schedule options must be ScheduleOptions, got "
            f"{type(options).__name__}"
        )
    return options


def schedule_cta(
    module: Module,
    function: Function,
    target: CudaTarget,
    topology: object,
    options: object | None = None,
) -> PipelineSchedulePlan:
    """Build, close, solve, and export the CTA pipeline schedule."""
    if function is not module.entry_function():
        raise ScheduleError(
            f"{TOPOLOGY} schedule requires the module entry function, got "
            f"{function.name!r}"
        )
    _options(options)
    program = build_pipeline_program(module, function)
    facts = target.as_facts(PipelineFacts, program.facts_query(TOPOLOGY))
    problem = build_pipeline_problem(program, facts, topology)
    solution = solve_pipeline_problem(problem)
    return export_pipeline_plan(program, solution, target)


register_schedule(CudaTarget, TOPOLOGY)(schedule_cta)


__all__ = ["TOPOLOGY", "schedule_cta"]
