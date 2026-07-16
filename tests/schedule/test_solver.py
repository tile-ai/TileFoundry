from __future__ import annotations

import math

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
