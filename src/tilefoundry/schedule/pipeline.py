from __future__ import annotations

from dataclasses import dataclass

from .builders.function_calls import FunctionCallGraphBuilder
from .cost import CostTable, build_cost_table
from .graph import ScheduleGraph
from .input import ScheduleInput
from .registry import ScheduleContext, resolve_schedule_backend
from .solution import ScheduleSolution
from .solver import SolveOptions, SolveProblem, problem_fingerprint
from .space import ScheduleSpace


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    graph: ScheduleGraph
    space: ScheduleSpace
    costs: CostTable
    solution: ScheduleSolution
    output: object


def run_schedule(
    schedule_input: ScheduleInput,
    context: ScheduleContext,
    options: SolveOptions | None = None,
) -> ScheduleResult:
    if not isinstance(schedule_input, ScheduleInput):
        raise TypeError("run_schedule expects a ScheduleInput")
    backend = resolve_schedule_backend(context.target, level=context.level)
    graph = FunctionCallGraphBuilder().build(schedule_input)
    space = backend.build_space(graph, context)
    costs = build_cost_table(space, backend.cost_model(context), context)
    fingerprint = problem_fingerprint(graph, space, costs, schedule_input.constraints)
    problem = SolveProblem(
        graph=graph,
        space=space,
        costs=costs,
        constraints=schedule_input.constraints,
        problem_fingerprint=fingerprint,
    )
    solution = backend.solver(context).solve(problem, options or SolveOptions())
    if solution.problem_fingerprint != fingerprint:
        raise ValueError("schedule solver returned a solution for a different problem")
    output = backend.materialize(problem, solution, context)
    return ScheduleResult(
        graph=graph,
        space=space,
        costs=costs,
        solution=solution,
        output=output,
    )


__all__ = ["ScheduleResult", "run_schedule"]
