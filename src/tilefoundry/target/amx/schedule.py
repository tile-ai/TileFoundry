"""The AMX scheduler registered at the core level."""

from __future__ import annotations

from tilefoundry.analysis import extract
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.schedule import ScheduleError, ScheduleOptions, ScheduleReport
from tilefoundry.schedule.kernel_schedule import build_schedule_tree
from tilefoundry.schedule.registry import register_schedule

# `select_atoms` measures its makespan in sub-ns duration units; the public
# report is in ns, and the scale that converts them has no public accessor yet.
from tilefoundry.schedule.select_atoms import _DURATION_SCALE, select_atoms

from .plan import CoreSchedulePlan
from .target import AmxTarget

TOPOLOGY = "core"


def schedule_core(
    module: Module,
    function: Function,
    target: AmxTarget,
    topology: object,
    options: object | None = None,
) -> CoreSchedulePlan:
    """Decide instructions and resources over *function*'s schedule tree.

    The decisions are made relative to the program's own entry, so this level
    schedules the entry function rather than any function of the Module.
    """
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
    solved = select_atoms(build_schedule_tree(extract(function)), target, TOPOLOGY)
    return CoreSchedulePlan(report=_project_report(function, target, solved))


def _project_report(
    function: Function, target: AmxTarget, solved: "TileGraph"
) -> ScheduleReport:
    """Project one resource-solve result into the compact objective summary.

    A solve that did not prove optimality leaves no bound behind, so the only
    one this level can state is the trivial zero.
    """
    makespan_ns = round(solved.decisions["makespan"] / _DURATION_SCALE)
    optimal = solved.decisions["status"] == "OPTIMAL"
    return ScheduleReport(
        root=function.name,
        target=target.name,
        topology=TOPOLOGY,
        status="OPTIMAL" if optimal else "FEASIBLE_NOT_PROVEN",
        objective_name="makespan",
        unit="ns",
        selected=makespan_ns,
        best_bound=makespan_ns if optimal else 0,
        gap=0.0 if optimal else 1.0,
    )


register_schedule(AmxTarget, TOPOLOGY)(schedule_core)


__all__ = ["TOPOLOGY", "schedule_core"]
