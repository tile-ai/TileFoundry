from __future__ import annotations

from dataclasses import dataclass

from .candidate import state_satisfies_constraints
from .solution import ScheduleSolution
from .solver import SolveProblem


@dataclass(frozen=True, slots=True)
class FunctionRegionReport:
    function_id: int
    name: str
    call_path: tuple[int, ...]
    op_count: int


@dataclass(frozen=True, slots=True)
class ConstraintReport:
    constraint_id: int
    kind: str
    target: object
    satisfied: bool


@dataclass(frozen=True, slots=True)
class OperationReport:
    op_id: int
    function_id: int
    call_path: tuple[int, ...]
    implementation_key: str
    cta_count: int
    submesh_offsets: tuple[int, ...]
    submesh_extents: tuple[int, ...]
    start_ns: int
    end_ns: int
    flops: int
    bytes: int
    compute_time_ns: int
    memory_time_ns: int
    duration_ns: int


@dataclass(frozen=True, slots=True)
class ReshardReport:
    edge_id: int
    moved_bytes: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True, slots=True)
class FusionOpportunity:
    edge_id: int
    source_op_id: int
    destination_op_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class ScheduleReport:
    status: str
    logical_fingerprint: str
    target: str
    level: str
    predicted_makespan_ns: int
    formula: str
    function_regions: tuple[FunctionRegionReport, ...]
    constraints: tuple[ConstraintReport, ...]
    operations: tuple[OperationReport, ...]
    reshards: tuple[ReshardReport, ...]
    critical_path: tuple[int, ...]
    unsupported_rules: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    fusion_opportunities: tuple[FusionOpportunity, ...] = ()

    def render_markdown(self) -> str:
        lines = [
            "# Schedule Report",
            "",
            f"- status: `{self.status}`",
            f"- logical fingerprint: `{self.logical_fingerprint}`",
            f"- target: `{self.target}`",
            f"- level: `{self.level}`",
            f"- predicted makespan (ns): `{self.predicted_makespan_ns}`",
            f"- cost model: `{self.formula}`",
            "",
            "## Functions",
            "",
            "| function id | name | call path | ops |",
            "| ---: | --- | --- | ---: |",
        ]
        lines.extend(
            f"| {item.function_id} | {item.name} | {item.call_path} | {item.op_count} |"
            for item in self.function_regions
        )
        lines.extend([
            "",
            "## Operations",
            "",
            "| op | function | call path | CTA count | submesh | start ns | end ns | duration ns |",
            "| ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |",
        ])
        lines.extend(
            f"| {item.op_id} | {item.function_id} | {item.call_path} | "
            f"{item.cta_count} | {item.submesh_offsets}/{item.submesh_extents} | "
            f"{item.start_ns} | {item.end_ns} | {item.duration_ns} |"
            for item in self.operations
        )
        lines.extend(["", "## Agent Constraints", "", "| id | kind | target | satisfied |", "| ---: | --- | --- | --- |"])
        lines.extend(
            f"| {item.constraint_id} | {item.kind} | `{item.target}` | {item.satisfied} |"
            for item in self.constraints
        )
        lines.extend(["", "## Reshards", "", "| edge | bytes | start ns | end ns |", "| ---: | ---: | ---: | ---: |"])
        lines.extend(
            f"| {item.edge_id} | {item.moved_bytes} | {item.start_ns} | {item.end_ns} |"
            for item in self.reshards
        )
        lines.extend(["", "## Critical Path", "", f"`{self.critical_path}`", ""])
        lines.extend(["## Fusion Opportunities", "", "| edge | source op | destination op | reason |", "| ---: | ---: | ---: | --- |"])
        lines.extend(
            f"| {item.edge_id} | {item.source_op_id} | {item.destination_op_id} | {item.reason} |"
            for item in self.fusion_opportunities
        )
        if self.unsupported_rules:
            lines.extend(["", "## Unsupported Rules", "", *[f"- {item}" for item in self.unsupported_rules]])
        if self.conflicts:
            lines.extend(["", "## Conflicts", "", *[f"- {item}" for item in self.conflicts]])
        return "\n".join(lines) + "\n"


def build_schedule_report(problem: SolveProblem, solution: ScheduleSolution, context) -> ScheduleReport:
    graph = problem.graph
    option_by_id = {option.id: option for option in problem.space.node_options}
    op_by_id = {op.id: op for op in graph.ops}
    function_regions = tuple(
        FunctionRegionReport(region.function_id, region.function.name, region.call_path, len(region.ops))
        for region in sorted(graph.regions, key=lambda item: (item.call_path, item.function_id))
    )
    constraints = tuple(
        ConstraintReport(
            item.id,
            type(item.constraint).__name__,
            item.target,
            state_satisfies_constraints(
                solution.value_for(item.target).state,
                (item.constraint,),
            ),
        )
        for item in graph.constraints
    )
    operations = []
    for assignment in sorted(solution.node_assignments, key=lambda item: item.node):
        op = op_by_id[assignment.node]
        option = option_by_id[assignment.option]
        estimate = problem.costs.node(assignment.option)
        operations.append(
            OperationReport(
                op_id=op.id,
                function_id=op.function_id,
                call_path=op.call_path,
                implementation_key=option.implementation_key,
                cta_count=option.candidate.cta_count,
                submesh_offsets=assignment.axis_starts,
                submesh_extents=assignment.axis_extents,
                start_ns=assignment.start_ns,
                end_ns=assignment.end_ns,
                flops=estimate.flops,
                bytes=estimate.traffic_bytes,
                compute_time_ns=estimate.compute_time_ns,
                memory_time_ns=estimate.memory_time_ns,
                duration_ns=estimate.duration_ns,
            )
        )
    reshards = tuple(
        ReshardReport(
            edge_id=edge.use,
            moved_bytes=problem.costs.edge(edge.option).traffic_bytes,
            start_ns=edge.start_ns,
            end_ns=edge.end_ns,
        )
        for edge in solution.edge_assignments
        if edge.kind.value == "reshard"
    )
    fusion = []
    for edge in graph.edges:
        try:
            use = solution.use_for(edge.id)
            source = solution.value_for(edge.source)
            destination = solution.value_for(edge.destination)
        except KeyError:
            continue
        if (
            use.kind.value != "direct"
            or use.source_state != use.destination_state
            or source.placement != destination.placement
        ):
            continue
        source_op = graph.value(edge.source).producer
        destination_op = edge.op_id
        if source_op is not None and destination_op is not None:
            fusion.append(FusionOpportunity(edge.id, source_op, destination_op, "identical selected representation and zero reshard"))
    operation_by_id = {item.op_id: item for item in operations}
    predecessor: dict[int, int] = {}
    predecessor_ready: dict[int, int] = {}
    for edge in graph.edges:
        source_op = graph.value(edge.source).producer
        if source_op is None or source_op not in operation_by_id:
            continue
        edge_assignment = solution.edge_for(edge.id)
        edge_duration = edge_assignment.end_ns - edge_assignment.start_ns
        if edge.kind == "call_arg":
            consumers = graph.value(edge.destination).consumers
        else:
            consumers = () if edge.op_id is None else (edge.op_id,)
        for consumer in consumers:
            ready = operation_by_id[source_op].end_ns + edge_duration
            if ready >= predecessor_ready.get(consumer, -1):
                predecessor_ready[consumer] = ready
                predecessor[consumer] = source_op
    if operation_by_id:
        current = max(operation_by_id.values(), key=lambda item: (item.end_ns, item.op_id)).op_id
        path = []
        seen = set()
        while current not in seen:
            seen.add(current)
            path.append(current)
            if current not in predecessor:
                break
            current = predecessor[current]
        critical_path = tuple(reversed(path))
    else:
        critical_path = ()
    target = repr(context.target)
    return ScheduleReport(
        status=solution.status,
        logical_fingerprint=graph.logical_fingerprint,
        target=target,
        level=context.level,
        predicted_makespan_ns=solution.makespan_ns,
        formula="uncalibrated formula: roofline plus zero fixed reshard latency",
        function_regions=function_regions,
        constraints=constraints,
        operations=tuple(operations),
        reshards=reshards,
        critical_path=critical_path,
        fusion_opportunities=tuple(fusion),
    )


__all__ = [
    "ConstraintReport",
    "FunctionRegionReport",
    "FusionOpportunity",
    "OperationReport",
    "ReshardReport",
    "ScheduleReport",
    "build_schedule_report",
]
