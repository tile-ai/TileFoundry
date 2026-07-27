"""Private public-report projection."""

from __future__ import annotations

from tilefoundry.schedule import ScheduleReport

from .planner import PlanningProblem


def project_schedule_report(
    problem: PlanningProblem, solution: "PlanningSolution", *, topology: str
) -> ScheduleReport:
    """Project one immutable planning solution into the compact public report."""
    return ScheduleReport(
        root=problem.root.name,
        target=problem.module.resolve_target().name,
        topology=topology,
        status=solution.status,
        objective_name="makespan",
        unit="ns",
        selected=solution.makespan_ns,
        best_bound=solution.best_bound_ns,
        gap=solution.gap,
    )


__all__ = ["project_schedule_report"]
