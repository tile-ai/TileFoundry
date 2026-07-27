"""The CUDA scheduler registered at the CTA level.

This is the whole of what `schedule(..., topology="cta")` does on CUDA: build the
private planning problem, solve it, materialize the per-CTA program, and state
the objective. The steps below the boundary stay private to this package.
"""

from __future__ import annotations

from pathlib import Path

from tilefoundry.inspection import as_script
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.tir.verify import verify_module
from tilefoundry.schedule import ScheduleError, ScheduleOptions
from tilefoundry.schedule.registry import register_schedule

from .materialize import materialize_planning_solution
from .plan import CtaSchedulePlan
from .planner import build_planning_problem
from .report import project_schedule_report
from .solver import solve_planning_problem
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
) -> CtaSchedulePlan:
    """Split *function* across the CTA hierarchy and materialize the result.

    The split is defined relative to the program's own entry, so this level
    schedules the entry function rather than any function of the Module.
    """
    if function is not module.entry_function():
        raise ScheduleError(
            f"{TOPOLOGY} schedule requires the module entry function, got "
            f"{function.name!r}"
        )
    resolved = _options(options)
    problem = build_planning_problem(module, function)
    solution = solve_planning_problem(problem, resolved)
    materialized = materialize_planning_solution(problem, solution)
    verify_module(materialized.functions)
    if resolved.debug_dump_dir is not None:
        _write_materialized_dump(materialized, resolved.debug_dump_dir)
    return CtaSchedulePlan(
        report=project_schedule_report(problem, solution, topology=TOPOLOGY),
        materialized=materialized,
    )


def _write_materialized_dump(module: Module, directory: Path) -> None:
    """Write the materialized program where a caller asked to inspect it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "materialized_hir.py").write_text(as_script(module))


register_schedule(CudaTarget, TOPOLOGY)(schedule_cta)


__all__ = ["TOPOLOGY", "schedule_cta"]
