from __future__ import annotations

import math
from dataclasses import replace

from tilefoundry.ir.target import CudaTarget
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.parser import parse_module_source
from tilefoundry.providers import resolve_provider_services
from tilefoundry.providers.cuda import CudaFormulaCostModel
from tilefoundry.schedule import (
    CpSatScheduleSolver,
    ScheduleContext,
    SolveOptions,
    SolveProblem,
    build_cost_table,
    build_program_schedule_graph,
    build_schedule_space,
    generate_distribution_candidates,
)
from tilefoundry.schedule.solver import problem_fingerprint
from tilefoundry.schedule.space import EdgeKind

SOURCE = '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Branches:
    @func
    def left_branch(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.add(x, x)

    @func
    def right_branch(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.add(x, x)

    @func
    def main(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        left = left_branch(x)
        right = right_branch(x)
        return tf.add(left, right)
'''


def _problem():
    module = parse_module_source(SOURCE)
    graph = build_program_schedule_graph(module)
    candidates = generate_distribution_candidates(graph, max_ctas=8)
    mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))
    target = CudaTarget(device="h200_sxm")
    services = resolve_provider_services(target, "cta")
    context = ScheduleContext(target, "cta", mesh, services)
    space = build_schedule_space(graph, candidates, parent_mesh=mesh)
    costs = build_cost_table(space, services.get(CudaFormulaCostModel), context)
    fingerprint = problem_fingerprint(graph, space, costs, graph.constraints)
    return graph, space, costs, fingerprint


def test_cost_table_is_finite_and_solver_fingerprint_matches() -> None:
    graph, space, costs, fingerprint = _problem()
    assert costs.all()
    assert all(math.isfinite(float(cost.duration_ns)) for cost in costs.all())

    solution = CpSatScheduleSolver().solve(
        SolveProblem(graph, space, costs, graph.constraints, fingerprint),
        SolveOptions(),
    )
    assert solution.problem_fingerprint == fingerprint
    assert solution.status == "OPTIMAL"


def test_independent_branches_overlap_on_disjoint_cta_slices() -> None:
    graph, space, costs, fingerprint = _problem()
    solution = CpSatScheduleSolver().solve(
        SolveProblem(graph, space, costs, graph.constraints, fingerprint),
        SolveOptions(),
    )
    branch_ops = [op for op in graph.ops if op.call_path in {(0,), (1,)}]
    assert len(branch_ops) == 2
    assignments = [solution.assignment_for(op.id) for op in branch_ops]
    assert len(assignments) == 2
    left, right = assignments
    assert left.start_ns < right.end_ns and right.start_ns < left.end_ns
    assert left.axis_starts[0] + left.axis_extents[0] <= right.axis_starts[0] or (
        right.axis_starts[0] + right.axis_extents[0] <= left.axis_starts[0]
    )

    selected_serial = sum(costs.node(assignment.option).duration_ns for assignment in solution.node_assignments)
    selected_serial += sum(costs.edge(edge.option).duration_ns for edge in solution.edge_assignments)
    assert solution.makespan_ns < selected_serial


def test_same_function_instances_share_scheme_but_not_placement() -> None:
    module = parse_module_source(
        '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class Shared:
    @func
    def leaf(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.add(x, x)

    @func
    def main(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        left = leaf(x)
        right = leaf(x)
        return tf.add(left, right)
'''
    )
    graph = build_program_schedule_graph(module)
    candidates = generate_distribution_candidates(graph, max_ctas=8)
    mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))
    target = CudaTarget(device="h200_sxm")
    services = resolve_provider_services(target, "cta")
    context = ScheduleContext(target, "cta", mesh, services)
    space = build_schedule_space(graph, candidates, parent_mesh=mesh)
    costs = build_cost_table(space, services.get(CudaFormulaCostModel), context)
    fingerprint = problem_fingerprint(graph, space, costs, graph.constraints)
    solution = CpSatScheduleSolver().solve(
        SolveProblem(graph, space, costs, graph.constraints, fingerprint),
        SolveOptions(),
    )

    leaf_function_id = next(
        region.function_id for region in graph.regions if region.function.name == "leaf"
    )
    leaf_ops = [op for op in graph.ops if op.function_id == leaf_function_id]
    assert len(leaf_ops) == 2
    assignments = [solution.assignment_for(op.id) for op in leaf_ops]
    options = {option.id: option for option in space.node_options}
    selected = [options[item.option].candidate for item in assignments]
    assert selected[0].input_states == selected[1].input_states
    assert selected[0].output_states == selected[1].output_states
    assert assignments[0].axis_starts != assignments[1].axis_starts
    assert assignments[0].start_ns < assignments[1].end_ns
    assert assignments[1].start_ns < assignments[0].end_ns


def test_call_argument_with_incompatible_states_requires_reshard() -> None:
    module = parse_module_source(
        '''from __future__ import annotations
from tilefoundry import module, func
from tilefoundry.dsl import Tensor, tf

@module(entry="main")
class CallBoundary:
    @func
    def leaf(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        return tf.add(x, x)

    @func
    def main(x: Tensor[(8,), "bf16"]) -> Tensor[(8,), "bf16"]:
        y = tf.add(x, x)
        return leaf(y)
'''
    )
    graph = build_program_schedule_graph(module)
    candidates = generate_distribution_candidates(graph, max_ctas=8)
    mesh = Mesh(Topology("cta", 8), Layout((8,), (1,)))
    target = CudaTarget(device="h200_sxm")
    services = resolve_provider_services(target, "cta")
    context = ScheduleContext(target, "cta", mesh, services)
    initial_space = build_schedule_space(graph, candidates, parent_mesh=mesh)
    forced_options = []
    for option in initial_space.node_options:
        if option.node in {0, 1, 2}:
            desired = {0: 8, 1: 4, 2: 4}[option.node]
            if option.candidate.cta_count != desired:
                continue
            placement = next(
                placement
                for placement in option.placements
                if placement.axis_extents == (desired,)
            )
            forced_options.append(replace(option, placements=(placement,)))
        else:
            forced_options.append(option)
    callee_param = next(
        value.ref
        for value in graph.values
        if value.ref.call_path == (0,) and value.producer is None
    )
    value_options = tuple(
        replace(
            option,
            states=tuple(state for state in option.states if state.cta_count == 4),
        )
        if option.value == callee_param
        else option
        for option in initial_space.value_options
    )
    space = replace(
        initial_space,
        node_options=tuple(forced_options),
        value_options=value_options,
    )
    costs = build_cost_table(space, services.get(CudaFormulaCostModel), context)
    fingerprint = problem_fingerprint(graph, space, costs, graph.constraints)
    solution = CpSatScheduleSolver().solve(
        SolveProblem(graph, space, costs, graph.constraints, fingerprint),
        SolveOptions(),
    )
    call_arg = next(edge for edge in graph.edges if edge.kind == "call_arg")
    assert solution.edge_for(call_arg.id).kind is EdgeKind.RESHARD
