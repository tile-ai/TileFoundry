from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from ortools.sat.python import cp_model

from .cost import CostTable
from .graph import GraphEdge, ProgramScheduleGraph
from .solution import EdgeAssignment, NodeAssignment, OpPlacement, ScheduleSolution
from .space import EdgeKind, NodeOption, PlacementOption, ScheduleSpace


@dataclass(frozen=True, slots=True)
class SolveOptions:
    deterministic: bool = True
    max_time_seconds: float | None = None


class ScheduleSolver(Protocol):
    def solve(self, problem: "SolveProblem", options: SolveOptions) -> ScheduleSolution:
        ...


class ScheduleInfeasibleError(ValueError):
    def __init__(self, message: str, *, problem_fingerprint: str) -> None:
        super().__init__(message)
        self.problem_fingerprint = problem_fingerprint


def _constraint_payload(constraint: object) -> object:
    if not dataclasses.is_dataclass(constraint):
        return repr(constraint)
    return {
        field.name: repr(getattr(constraint, field.name))
        for field in dataclasses.fields(constraint)
        if field.name not in {"source_loc"}
    }


def problem_fingerprint(
    graph: ProgramScheduleGraph,
    space: ScheduleSpace,
    costs: CostTable,
    constraints: tuple[object, ...],
) -> str:
    payload = {
        "logical": graph.logical_fingerprint,
        "nodes": [
            {
                "id": option.id,
                "node": option.node,
                "candidate": repr(option.candidate),
                "placements": [
                    (placement.axis_starts, placement.axis_extents)
                    for placement in option.placements
                ],
            }
            for option in space.node_options
        ],
        "edges": [
            (option.id, option.use, option.kind.value, option.payload_bytes, option.cta_count)
            for option in space.edge_options
        ],
        "node_costs": [(key, repr(value)) for key, value in costs.node_costs],
        "edge_costs": [(key, repr(value)) for key, value in costs.edge_costs],
        "constraints": [repr(item.constraint) for item in constraints],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SolveProblem:
    graph: ProgramScheduleGraph
    space: ScheduleSpace
    costs: CostTable
    constraints: tuple[object, ...]
    problem_fingerprint: str


@dataclass(frozen=True, slots=True)
class _Choice:
    node: NodeOption
    placement: PlacementOption
    variable: cp_model.IntVar


def _states_equal(left, right) -> bool:
    return left == right


class CpSatScheduleSolver:
    """Common one-dimensional CTA CP-SAT solver."""

    def solve(self, problem: SolveProblem, options: SolveOptions) -> ScheduleSolution:
        graph = problem.graph
        space = problem.space
        costs = problem.costs
        model = cp_model.CpModel()
        node_choices: dict[int, list[_Choice]] = defaultdict(list)
        start_vars: dict[int, cp_model.IntVar] = {}
        end_vars: dict[int, cp_model.IntVar] = {}
        x_intervals = []
        y_intervals = []

        max_node_cost = max((costs.node(option.id).duration_ns for option in space.node_options), default=1)
        max_edge_cost = max((costs.edge(option.id).duration_ns for option in space.edge_options), default=0)
        horizon = max(1, (max_node_cost + max_edge_cost) * max(1, len(graph.ops) * 2))
        for op in graph.ops:
            start = model.NewIntVar(0, horizon, f"start_{op.id}")
            end = model.NewIntVar(0, horizon, f"end_{op.id}")
            start_vars[op.id] = start
            end_vars[op.id] = end
            options_for_node = space.options_for_node(op.id)
            if not options_for_node:
                raise ScheduleInfeasibleError(
                    f"no legal candidate for graph op {op.id}",
                    problem_fingerprint=problem.problem_fingerprint,
                )
            for node_option in options_for_node:
                duration = costs.node(node_option.id).duration_ns
                for placement in node_option.placements:
                    present = model.NewBoolVar(f"op_{op.id}_option_{node_option.id}_place_{placement.id}")
                    model.Add(end == start + duration).OnlyEnforceIf(present)
                    x_intervals.append(
                        model.NewOptionalIntervalVar(
                            start,
                            duration,
                            end,
                            present,
                            f"time_{op.id}_{placement.id}",
                        )
                    )
                    y_start = model.NewConstant(placement.axis_starts[0])
                    y_end = model.NewConstant(placement.axis_starts[0] + placement.axis_extents[0])
                    y_intervals.append(
                        model.NewOptionalIntervalVar(
                            y_start,
                            placement.axis_extents[0],
                            y_end,
                            present,
                            f"space_{op.id}_{placement.id}",
                        )
                    )
                    node_choices[op.id].append(_Choice(node_option, placement, present))
            model.Add(sum(choice.variable for choice in node_choices[op.id]) == 1)

        model.AddNoOverlap2D(x_intervals, y_intervals)
        self._constrain_shared_function_schemes(model, graph, node_choices)

        edge_variables: dict[int, dict[EdgeKind, cp_model.IntVar]] = {}
        for edge in graph.edges:
            edge_options = space.options_for_use(edge.id)
            if not edge_options:
                continue
            by_kind: dict[EdgeKind, cp_model.IntVar] = {}
            for edge_option in edge_options:
                by_kind[edge_option.kind] = model.NewBoolVar(
                    f"edge_{edge.id}_{edge_option.kind.value}"
                )
            edge_variables[edge.id] = by_kind
            model.Add(sum(by_kind.values()) == 1)
            producer = graph.value(edge.source).producer
            if producer is None:
                direct = by_kind.get(EdgeKind.DIRECT)
                if direct is not None:
                    model.Add(direct == 1)
                continue
            consumer = edge.op_id if edge.kind != "call_arg" else None
            if consumer is None:
                continue
            direct = by_kind.get(EdgeKind.DIRECT)
            reshard = by_kind.get(EdgeKind.RESHARD)
            if direct is None or reshard is None:
                continue
            self._constrain_edge_compatibility(
                model,
                graph,
                edge,
                direct,
                node_choices[producer],
                node_choices[consumer],
            )
            reshard_cost = next(
                costs.edge(option.id).duration_ns
                for option in edge_options
                if option.kind is EdgeKind.RESHARD
            )
            model.Add(start_vars[consumer] >= end_vars[producer]).OnlyEnforceIf(direct)
            model.Add(start_vars[consumer] >= end_vars[producer] + reshard_cost).OnlyEnforceIf(reshard)

        self._constrain_call_argument_dependencies(
            model, graph, space, edge_variables, start_vars, end_vars, costs
        )
        makespan = model.NewIntVar(0, horizon, "makespan")
        for end in end_vars.values():
            model.Add(makespan >= end)
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1 if options.deterministic else 8
        solver.parameters.random_seed = 0
        if options.max_time_seconds is not None:
            solver.parameters.max_time_in_seconds = options.max_time_seconds
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise ScheduleInfeasibleError(
                f"schedule is infeasible (CP-SAT status {solver.StatusName(status)})",
                problem_fingerprint=problem.problem_fingerprint,
            )

        node_assignments = []
        for op in graph.ops:
            selected = next(choice for choice in node_choices[op.id] if solver.Value(choice.variable))
            node_assignments.append(
                NodeAssignment(
                    node=op.id,
                    candidate=selected.node.candidate.id,
                    option=selected.node.id,
                    placement=OpPlacement(
                        start_ns=solver.Value(start_vars[op.id]),
                        end_ns=solver.Value(end_vars[op.id]),
                        submesh=selected.placement.submesh,
                    ),
                )
            )

        node_by_id = {assignment.node: assignment for assignment in node_assignments}
        edge_assignments = []
        for edge in graph.edges:
            variables = edge_variables.get(edge.id)
            if variables is None:
                continue
            selected_kind = next(kind for kind, variable in variables.items() if solver.Value(variable))
            selected_option = next(
                option
                for option in space.options_for_use(edge.id)
                if option.kind is selected_kind
            )
            producer = graph.value(edge.source).producer
            start = 0 if producer is None else node_by_id[producer].end_ns
            duration = costs.edge(selected_option.id).duration_ns
            edge_assignments.append(
                EdgeAssignment(edge.id, selected_option.id, selected_kind, start, start + duration)
            )
        status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE_NOT_PROVEN"
        return ScheduleSolution(
            node_assignments=tuple(node_assignments),
            edge_assignments=tuple(edge_assignments),
            makespan_ns=solver.Value(makespan),
            problem_fingerprint=problem.problem_fingerprint,
            status=status_name,
        )

    @staticmethod
    def _constrain_shared_function_schemes(model, graph, node_choices) -> None:
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for op in graph.ops:
            groups[(op.function_id, id(op.ir_expr))].append(op.id)
        for op_ids in groups.values():
            if len(op_ids) < 2:
                continue
            signatures = set()
            by_op: dict[int, dict[object, list[cp_model.IntVar]]] = {}
            for op_id in op_ids:
                by_op[op_id] = defaultdict(list)
                for choice in node_choices[op_id]:
                    signature = (
                        repr(choice.node.candidate.input_states),
                        repr(choice.node.candidate.output_states),
                        choice.placement.axis_starts,
                        choice.placement.axis_extents,
                    )
                    signatures.add(signature)
                    by_op[op_id][signature].append(choice.variable)
            first = op_ids[0]
            for signature in signatures:
                first_sum = sum(by_op[first].get(signature, ()))
                for op_id in op_ids[1:]:
                    model.Add(sum(by_op[op_id].get(signature, ())) == first_sum)

    @staticmethod
    def _constrain_edge_compatibility(
        model,
        graph,
        edge: GraphEdge,
        direct,
        producer_choices: list[_Choice],
        consumer_choices: list[_Choice],
    ) -> None:
        for producer in producer_choices:
            for consumer in consumer_choices:
                if edge.kind == "call_result":
                    compatible = _states_equal(
                        producer.node.candidate.output_states[0],
                        consumer.node.candidate.output_states[0],
                    )
                else:
                    index = edge.operand_index or 0
                    inputs = consumer.node.candidate.input_states
                    compatible = index < len(inputs) and _states_equal(
                        producer.node.candidate.output_states[0], inputs[index]
                    )
                if edge.kind == "data" and compatible:
                    compatible = (
                        producer.placement.axis_starts == consumer.placement.axis_starts
                        and producer.placement.axis_extents == consumer.placement.axis_extents
                    )
                if not compatible:
                    model.AddBoolOr([
                        producer.variable.Not(),
                        consumer.variable.Not(),
                        direct.Not(),
                    ])

    @staticmethod
    def _constrain_call_argument_dependencies(
        model,
        graph,
        space,
        edge_variables,
        start_vars,
        end_vars,
        costs,
    ) -> None:
        values = {value.ref: value for value in graph.values}
        for edge in graph.edges:
            if edge.kind != "call_arg":
                continue
            producer = values[edge.source].producer
            if producer is None:
                continue
            for consumer in values[edge.destination].consumers:
                variables = edge_variables.get(edge.id, {})
                direct = variables.get(EdgeKind.DIRECT)
                reshard = variables.get(EdgeKind.RESHARD)
                if direct is not None:
                    model.Add(start_vars[consumer] >= end_vars[producer]).OnlyEnforceIf(direct)
                if reshard is not None:
                    option = next(
                        option
                        for option in space.options_for_use(edge.id)
                        if option.kind is EdgeKind.RESHARD
                    )
                    model.Add(
                        start_vars[consumer]
                        >= end_vars[producer] + costs.edge(option.id).duration_ns
                    ).OnlyEnforceIf(reshard)
__all__ = [
    "CpSatScheduleSolver",
    "ScheduleInfeasibleError",
    "ScheduleSolver",
    "SolveOptions",
    "SolveProblem",
    "problem_fingerprint",
]
