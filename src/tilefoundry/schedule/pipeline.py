from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types.shard import Layout, Mesh, Topology
from tilefoundry.providers import resolve_provider_services
from tilefoundry.providers.services import TargetScheduleProfile

from .candidate import generate_distribution_candidates
from .context import ScheduleContext
from .cost import build_cost_table
from .graph import ProgramScheduleGraph, build_program_schedule_graph
from .materialize import materialize_schedule
from .solver import CpSatScheduleSolver, SolveOptions, SolveProblem, problem_fingerprint
from .space import ScheduleSpace, build_schedule_space


@dataclass(frozen=True, slots=True)
class SolveResult:
    solution: Module
    report: object | None
    graph: ProgramScheduleGraph
    space: ScheduleSpace
    costs: object


def _as_module(candidate: Module | Function) -> Module:
    if isinstance(candidate, Module):
        return candidate
    if isinstance(candidate, Function):
        return Module(candidate.name, (candidate,), candidate.name, candidate.topologies)
    raise TypeError(f"auto_dist expects a HIR Module or Function, got {type(candidate).__name__}")


def _service_by_capability(services: object, method: str) -> object:
    for _, service in getattr(services, "services", ()):
        if callable(getattr(service, method, None)):
            return service
    raise ValueError(f"provider did not register a service with {method}()")


def _default_mesh(profile: TargetScheduleProfile) -> Mesh:
    extent = profile.max_ctas
    return Mesh(
        Topology(profile.topology, extent),
        Layout((extent,), (1,)),
    )


def auto_dist(
    candidate: Module | Function,
    context: ScheduleContext | None = None,
    *,
    target: object | None = None,
    level: str = "cta",
    mesh: object | None = None,
    options: SolveOptions | None = None,
) -> SolveResult:
    """Solve one complete HIR Module through the target-provider services."""
    module = _as_module(candidate)
    if context is None:
        if target is None:
            raise ValueError("auto_dist requires a concrete target or ScheduleContext")
        context = ScheduleContext(target=target, level=level, mesh=mesh)
    elif target is not None or mesh is not None:
        raise TypeError("auto_dist accepts either context or target/mesh keyword arguments")

    services = context.services or resolve_provider_services(context.target, context.level)
    profile = services.get(TargetScheduleProfile)
    parent_mesh = context.mesh or _default_mesh(profile)
    context = context.with_services(services)
    if context.mesh is None:
        context = ScheduleContext(context.target, context.level, parent_mesh, services)

    graph = build_program_schedule_graph(module)
    candidates = generate_distribution_candidates(
        graph,
        max_ctas=min(profile.max_ctas, parent_mesh.shape[0]),
    )
    space = build_schedule_space(graph, candidates, parent_mesh=parent_mesh)
    model = _service_by_capability(services, "estimate_node")
    costs = build_cost_table(space, model, context)
    fingerprint = problem_fingerprint(graph, space, costs, graph.constraints)
    problem = SolveProblem(graph, space, costs, graph.constraints, fingerprint)
    solution = CpSatScheduleSolver().solve(problem, options or SolveOptions())
    materialized = materialize_schedule(problem, solution, context)
    return SolveResult(
        solution=materialized,
        report=None,
        graph=graph,
        space=space,
        costs=costs,
    )


__all__ = ["SolveResult", "auto_dist"]
