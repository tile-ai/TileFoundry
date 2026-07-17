from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

from ortools.sat.python import cp_model

from .candidate import DistributionState, LayoutState, Submesh
from .cost import CostTable
from .graph import GraphEdge, ProgramScheduleGraph
from .solution import (
    EdgeAssignment,
    NodeAssignment,
    OpPlacement,
    ScheduleSolution,
    UseAssignment,
    ValueAssignment,
)
from .space import EdgeKind, NodeOption, PlacementOption, ScheduleSpace


@dataclass(frozen=True, slots=True)
class SolveOptions:
    deterministic: bool = True
    max_time_seconds: float | None = 10.0


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


@dataclass(frozen=True, slots=True)
class _StateChoice:
    state: DistributionState
    variable: cp_model.IntVar


def _states_compatible(source: DistributionState, destination: DistributionState) -> bool:
    if source == destination:
        return True
    if source.partial is not None or destination.partial is not None:
        return False
    return source.layout.is_broadcast and source.cta_count == 1


def _placements_compatible(source_state, source_placement, destination_placement) -> bool:
    if source_placement is None or destination_placement is None:
        return True
    source_starts = getattr(source_placement, "axis_starts", None)
    if source_starts is None:
        source_starts = source_placement.offsets
    source_extents = getattr(source_placement, "axis_extents", None)
    if source_extents is None:
        source_extents = source_placement.extents
    destination_starts = getattr(destination_placement, "axis_starts", None)
    if destination_starts is None:
        destination_starts = destination_placement.offsets
    destination_extents = getattr(destination_placement, "axis_extents", None)
    if destination_extents is None:
        destination_extents = destination_placement.extents
    if source_state.layout.is_broadcast and source_state.cta_count == 1:
        source_start = source_starts[0]
        source_end = source_start + source_extents[0]
        destination_start = destination_starts[0]
        destination_end = destination_start + destination_extents[0]
        return source_start <= destination_start and destination_end <= source_end
    return (
        source_starts == destination_starts
        and source_extents == destination_extents
    )


class CpSatScheduleSolver:
    """Common one-dimensional CTA CP-SAT solver."""

    def solve(self, problem: SolveProblem, options: SolveOptions) -> ScheduleSolution:
        graph = problem.graph
        space = problem.space
        costs = problem.costs
        model = cp_model.CpModel()
        node_choices: dict[int, list[_Choice]] = defaultdict(list)
        value_choices: dict[object, list[_StateChoice]] = defaultdict(list)
        start_vars: dict[int, cp_model.IntVar] = {}
        end_vars: dict[int, cp_model.IntVar] = {}
        x_intervals = []
        y_intervals = []
        edge_cost_terms = []

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

        for value_option in space.value_options:
            for index, state in enumerate(value_option.states):
                variable = model.NewBoolVar(
                    f"value_{value_option.value.function_id}_"
                    f"{value_option.value.local_value_id}_{index}"
                )
                value_choices[value_option.value].append(_StateChoice(state, variable))
            model.Add(sum(choice.variable for choice in value_choices[value_option.value]) == 1)

        for value in graph.values:
            if value.producer is not None or value.ref in value_choices:
                continue
            state = DistributionState(LayoutState(len(getattr(value.ir_value.type, "shape", ()))))
            variable = model.NewBoolVar(
                f"value_{value.ref.function_id}_{value.ref.local_value_id}_default"
            )
            value_choices[value.ref].append(_StateChoice(state, variable))
            model.Add(variable == 1)

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
            direct = by_kind.get(EdgeKind.DIRECT)
            reshard = by_kind.get(EdgeKind.RESHARD)
            if direct is not None:
                source_choices = self._source_state_choices(
                    graph, edge.source, node_choices, value_choices
                )
                destination_choices = self._destination_state_choices(
                    graph, edge, node_choices, value_choices
                )
                for source_state, source_variables, source_placement in source_choices:
                    for destination_state, destination_variables, destination_placement in destination_choices:
                        compatible = _states_compatible(source_state, destination_state)
                        if compatible and edge.kind in {"data", "call_arg", "call_result"}:
                            compatible = _placements_compatible(
                                source_state,
                                source_placement,
                                destination_placement,
                            )
                        if not compatible:
                            model.AddBoolOr([
                                *(variable.Not() for variable in source_variables),
                                *(variable.Not() for variable in destination_variables),
                                direct.Not(),
                            ])

            reshard_cost = next(
                costs.edge(option.id).duration_ns
                for option in edge_options
                if option.kind is EdgeKind.RESHARD
            ) if reshard is not None else 0
            if reshard is not None:
                edge_cost_terms.append(reshard_cost * reshard)
            producer = graph.value(edge.source).producer
            if producer is None or edge.op_id is None:
                continue
            if edge.kind == "call_arg":
                consumers = graph.value(edge.destination).consumers
            else:
                consumers = (edge.op_id,)
            for consumer in consumers:
                if direct is not None:
                    model.Add(start_vars[consumer] >= end_vars[producer]).OnlyEnforceIf(direct)
                if reshard is not None:
                    model.Add(
                        start_vars[consumer] >= end_vars[producer] + reshard_cost
                    ).OnlyEnforceIf(reshard)

        makespan = model.NewIntVar(0, horizon, "makespan")
        for end in end_vars.values():
            model.Add(makespan >= end)
        edge_cost_bound = max_edge_cost * max(1, len(graph.edges))
        start_cost_scale = edge_cost_bound + 1
        start_cost_bound = horizon * max(1, len(graph.ops))
        makespan_scale = start_cost_bound * start_cost_scale + edge_cost_bound + 1
        model.Minimize(
            makespan * makespan_scale
            + sum(start_vars.values()) * start_cost_scale
            + sum(edge_cost_terms)
        )

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1 if options.deterministic else 8
        solver.parameters.random_seed = 0
        if options.max_time_seconds is not None:
            solver.parameters.max_time_in_seconds = options.max_time_seconds
        status = solver.Solve(model)
        if status == cp_model.UNKNOWN:
            return self._fallback_solution(problem)
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
        selected_value_states = {
            value_ref: next(
                choice.state
                for choice in choices
                if solver.Value(choice.variable)
            )
            for value_ref, choices in value_choices.items()
        }
        status_name = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE_NOT_PROVEN"
        return self._complete_solution(
            problem,
            tuple(node_assignments),
            tuple(edge_assignments),
            selected_value_states,
            solver.Value(makespan),
            status_name,
        )

    @staticmethod
    def _complete_solution(
        problem: SolveProblem,
        node_assignments: tuple[NodeAssignment, ...],
        edge_assignments: tuple[EdgeAssignment, ...],
        selected_value_states: dict[object, DistributionState],
        makespan: int,
        status: str,
    ) -> ScheduleSolution:
        graph = problem.graph
        space = problem.space
        node_by_id = {assignment.node: assignment for assignment in node_assignments}
        option_by_id = {option.id: option for option in space.node_options}
        edge_assignment_by_id = {
            assignment.use: assignment for assignment in edge_assignments
        }
        parent_extent = space.resources[0].capacity if space.resources else 1
        full_parent = Submesh((0,), (parent_extent,))

        def value_state(value_ref):
            value = graph.value(value_ref)
            if value.producer is not None:
                assignment = node_by_id[value.producer]
                return option_by_id[assignment.option].candidate.output_states[0]
            if value_ref in selected_value_states:
                return selected_value_states[value_ref]
            try:
                return space.options_for_value(value_ref).states[0]
            except KeyError:
                return DistributionState(LayoutState(len(getattr(value.ir_value.type, "shape", ()))))

        def value_placement(value_ref):
            value = graph.value(value_ref)
            if value.producer is not None:
                return node_by_id[value.producer].placement.submesh
            state = value_state(value_ref)
            if value_ref.call_path == () and state.layout.is_broadcast:
                return full_parent
            for edge in graph.edges:
                if edge.destination != value_ref or edge.kind != "call_arg":
                    continue
                source = graph.value(edge.source)
                source_assignment = (
                    None
                    if source.producer is None
                    else node_by_id[source.producer]
                )
                if source_assignment is not None:
                    edge_assignment = edge_assignment_by_id.get(edge.id)
                    if edge_assignment is not None and edge_assignment.kind is EdgeKind.DIRECT:
                        return source_assignment.placement.submesh
                consumers = graph.value(edge.destination).consumers
                for consumer in consumers:
                    if consumer in node_by_id:
                        return node_by_id[consumer].placement.submesh
            for edge in graph.edges:
                if edge.source == value_ref and edge.op_id in node_by_id:
                    return node_by_id[edge.op_id].placement.submesh
            return full_parent

        value_assignments = tuple(
            ValueAssignment(value.ref, value_state(value.ref), value_placement(value.ref))
            for value in graph.values
        )
        value_assignment_by_ref = {
            assignment.value: assignment for assignment in value_assignments
        }

        def destination_state(edge):
            if edge.kind == "call_arg":
                return value_assignment_by_ref[edge.destination].state
            assignment = node_by_id[edge.op_id]
            option = option_by_id[assignment.option]
            if edge.kind == "call_result":
                return option.candidate.output_states[0]
            return option.candidate.input_states[edge.operand_index or 0]

        use_assignments = tuple(
            UseAssignment(
                edge_id=edge.id,
                kind=edge_assignment_by_id[edge.id].kind,
                source_state=value_assignment_by_ref[edge.source].state,
                destination_state=destination_state(edge),
                moved_bytes=problem.costs.edge(
                    edge_assignment_by_id[edge.id].option
                ).traffic_bytes,
            )
            for edge in graph.edges
            if edge.id in edge_assignment_by_id
        )
        return ScheduleSolution(
            node_assignments=node_assignments,
            edge_assignments=edge_assignments,
            makespan_ns=makespan,
            problem_fingerprint=problem.problem_fingerprint,
            status=status,
            value_assignments=value_assignments,
            use_assignments=use_assignments,
        )

    @staticmethod
    def _fallback_solution(problem: SolveProblem) -> ScheduleSolution:
        """Return a deterministic serial incumbent when CP-SAT times out first."""
        graph = problem.graph
        space = problem.space
        costs = problem.costs
        predecessors: dict[int, list[tuple[int, int]]] = defaultdict(list)
        initial_ready: dict[int, int] = defaultdict(int)
        for edge in graph.edges:
            options = space.options_for_use(edge.id)
            if not options:
                continue
            producer = graph.value(edge.source).producer
            if producer is None:
                reshard = next(option for option in options if option.kind is EdgeKind.RESHARD)
                if edge.kind == "call_arg":
                    consumers = graph.value(edge.destination).consumers
                elif edge.op_id is not None:
                    consumers = (edge.op_id,)
                else:
                    consumers = ()
                delay = costs.edge(reshard.id).duration_ns
                for consumer in consumers:
                    initial_ready[consumer] = max(initial_ready[consumer], delay)
                continue
            reshard = next(option for option in options if option.kind is EdgeKind.RESHARD)
            if edge.kind == "call_arg":
                consumers = graph.value(edge.destination).consumers
            elif edge.op_id is not None:
                consumers = (edge.op_id,)
            else:
                consumers = ()
            for consumer in consumers:
                predecessors[consumer].append((producer, costs.edge(reshard.id).duration_ns))

        order: list[int] = []
        remaining = {op.id for op in graph.ops}
        while remaining:
            ready = sorted(
                op_id
                for op_id in remaining
                if all(producer in order for producer, _ in predecessors.get(op_id, ()))
            )
            if not ready:
                raise ScheduleInfeasibleError(
                    "schedule dependency graph is cyclic",
                    problem_fingerprint=problem.problem_fingerprint,
                )
            order.extend(ready)
            remaining.difference_update(ready)

        assignments_by_id: dict[int, NodeAssignment] = {}
        for op_id in order:
            start = max(
                [initial_ready.get(op_id, 0)]
                + [
                    assignments_by_id[producer].end_ns + edge_duration
                    for producer, edge_duration in predecessors.get(op_id, ())
                ]
            )
            selected_option = None
            selected_placement = None
            selected_end = None
            while selected_option is None:
                for option in space.options_for_node(op_id):
                    duration = costs.node(option.id).duration_ns
                    for placement in option.placements:
                        end = start + duration
                        conflicts = [
                            prior
                            for prior in assignments_by_id.values()
                            if start < prior.end_ns
                            and prior.start_ns < end
                            and placement.axis_starts[0] < prior.axis_starts[0] + prior.axis_extents[0]
                            and prior.axis_starts[0] < placement.axis_starts[0] + placement.axis_extents[0]
                        ]
                        if not conflicts:
                            selected_option = option
                            selected_placement = placement
                            selected_end = end
                            break
                    if selected_option is not None:
                        break
                if selected_option is None:
                    blocking = [
                        prior.end_ns
                        for prior in assignments_by_id.values()
                        if prior.end_ns > start
                    ]
                    if not blocking:
                        raise ScheduleInfeasibleError(
                            f"no finite placement for graph op {op_id}",
                            problem_fingerprint=problem.problem_fingerprint,
                        )
                    start = min(blocking)
            assert selected_placement is not None and selected_end is not None
            assignments_by_id[op_id] = NodeAssignment(
                node=op_id,
                candidate=selected_option.candidate.id,
                option=selected_option.id,
                placement=OpPlacement(start, selected_end, selected_placement.submesh),
            )

        option_by_id = {option.id: option for option in space.node_options}
        selected_value_states = {
            value_option.value: value_option.states[0]
            for value_option in space.value_options
        }
        full_parent = Submesh((0,), (space.resources[0].capacity,))

        def value_state(value_ref):
            value = graph.value(value_ref)
            if value.producer is not None:
                assignment = assignments_by_id[value.producer]
                return option_by_id[assignment.option].candidate.output_states[0]
            return selected_value_states.get(
                value_ref,
                DistributionState(LayoutState(len(getattr(value.ir_value.type, "shape", ())))),
            )

        def value_placement(value_ref):
            value = graph.value(value_ref)
            if value.producer is not None:
                return assignments_by_id[value.producer].placement.submesh
            state = value_state(value_ref)
            if value_ref.call_path == () and state.layout.is_broadcast:
                return full_parent
            for consumer in value.consumers:
                assignment = assignments_by_id.get(consumer)
                if assignment is not None:
                    return assignment.placement.submesh
            return full_parent

        def destination_state(edge):
            if edge.kind == "call_arg":
                return value_state(edge.destination)
            assignment = assignments_by_id[edge.op_id]
            option = option_by_id[assignment.option]
            if edge.kind == "call_result":
                return option.candidate.output_states[0]
            return option.candidate.input_states[edge.operand_index or 0]

        def destination_placement(edge):
            if edge.kind == "call_arg":
                return value_placement(edge.destination)
            return assignments_by_id[edge.op_id].placement.submesh

        selected_edge_choices: dict[int, EdgeOption] = {}
        for edge in graph.edges:
            options = space.options_for_use(edge.id)
            if not options:
                continue
            direct = next(
                (option for option in options if option.kind is EdgeKind.DIRECT),
                None,
            )
            reshard = next(
                (option for option in options if option.kind is EdgeKind.RESHARD),
                None,
            )
            if direct is None:
                selected_edge_choices[edge.id] = reshard
                continue
            direct_allowed = _states_compatible(
                value_state(edge.source),
                destination_state(edge),
            ) and _placements_compatible(
                value_state(edge.source),
                value_placement(edge.source),
                destination_placement(edge),
            )
            selected_edge_choices[edge.id] = direct if direct_allowed else reshard

        edge_assignments = []
        for edge_id, option in sorted(selected_edge_choices.items()):
            edge = next(edge for edge in graph.edges if edge.id == edge_id)
            producer = graph.value(edge.source).producer
            start = 0 if producer is None else assignments_by_id[producer].end_ns
            duration = costs.edge(option.id).duration_ns
            edge_assignments.append(EdgeAssignment(edge_id, option.id, option.kind, start, start + duration))
        makespan = max((assignment.end_ns for assignment in assignments_by_id.values()), default=0)
        return CpSatScheduleSolver._complete_solution(
            problem,
            tuple(assignments_by_id[op.id] for op in graph.ops),
            tuple(edge_assignments),
            selected_value_states,
            makespan,
            "FEASIBLE_NOT_PROVEN",
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
                        choice.node.implementation_key,
                        repr(choice.node.candidate.input_states),
                        repr(choice.node.candidate.output_states),
                    )
                    signatures.add(signature)
                    by_op[op_id][signature].append(choice.variable)
            first = op_ids[0]
            for signature in signatures:
                first_sum = sum(by_op[first].get(signature, ()))
                for op_id in op_ids[1:]:
                    model.Add(sum(by_op[op_id].get(signature, ())) == first_sum)

    @staticmethod
    def _source_state_choices(graph, value_ref, node_choices, value_choices):
        value = graph.value(value_ref)
        if value.producer is None:
            consumers = value.consumers
            choices = []
            for value_choice in value_choices[value_ref]:
                if value_ref.call_path == () and value_choice.state.layout.is_broadcast:
                    choices.append((value_choice.state, (value_choice.variable,), None))
                    continue
                if not consumers:
                    choices.append((value_choice.state, (value_choice.variable,), None))
                    continue
                for consumer in consumers:
                    for node_choice in node_choices[consumer]:
                        choices.append(
                            (
                                value_choice.state,
                                (value_choice.variable, node_choice.variable),
                                node_choice.placement,
                            )
                        )
            return tuple(choices)
        return tuple(
            (
                choice.node.candidate.output_states[0],
                (choice.variable,),
                choice.placement,
            )
            for choice in node_choices[value.producer]
        )

    @staticmethod
    def _destination_state_choices(graph, edge, node_choices, value_choices):
        if edge.kind == "call_arg":
            consumers = graph.value(edge.destination).consumers
            choices = []
            for value_choice in value_choices[edge.destination]:
                if not consumers:
                    choices.append((value_choice.state, (value_choice.variable,), None))
                    continue
                for consumer in consumers:
                    for node_choice in node_choices[consumer]:
                        choices.append(
                            (
                                value_choice.state,
                                (value_choice.variable, node_choice.variable),
                                node_choice.placement,
                            )
                        )
            return tuple(choices)
        if edge.op_id is None:
            return ()
        choices = []
        for choice in node_choices[edge.op_id]:
            if edge.kind == "call_result":
                state = choice.node.candidate.output_states[0]
            else:
                index = edge.operand_index or 0
                if index >= len(choice.node.candidate.input_states):
                    continue
                state = choice.node.candidate.input_states[index]
            choices.append((state, (choice.variable,), choice.placement))
        return tuple(choices)

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
