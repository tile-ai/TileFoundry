"""Typed export of a solved pipeline schedule."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass

from tilefoundry.schedule.plan import PlanVerificationError, SchedulePlan

from .program import PipelineProgram
from .solve import PipelineSolution


@dataclass(frozen=True)
class TargetSpecRef:
    """Stable identity of the installed target facts a plan relies on."""

    architecture_id: str
    architecture_digest: str
    device_id: str
    device_digest: str


@dataclass(frozen=True)
class ScheduledStatement:
    """One selected instruction and its half-open execution interval."""

    id: str
    instruction: str
    tile: tuple[int, ...]
    resources: tuple[tuple[str, int], ...]
    start: int
    end: int


@dataclass(frozen=True)
class ScheduledBuffer:
    """One named storage object and its dependency-safe ring allocation."""

    id: str
    storage: str
    ring_depth: int
    producer_ids: tuple[str, ...]
    consumer_ids: tuple[str, ...]


@dataclass(frozen=True)
class KernelHole:
    """One stable statement reference with serialized input/output relations."""

    statement_id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    relations: tuple[str, ...]


@dataclass(frozen=True)
class PipelineSchedulePlan(SchedulePlan):
    """The complete, deterministic exported result of a CTA pipeline solve."""

    target: TargetSpecRef
    scaffold: str
    statements: tuple[ScheduledStatement, ...]
    buffers: tuple[ScheduledBuffer, ...]
    holes: tuple[KernelHole, ...]

    def verify(self, module, function, topology) -> None:
        statement_ids = {item.id for item in self.statements}
        if len(statement_ids) != len(self.statements):
            raise PlanVerificationError("pipeline plan has duplicate statement IDs")
        for item in self.statements:
            if item.start < 0 or item.end <= item.start:
                raise PlanVerificationError(f"statement {item.id!r} has invalid interval")
        for buffer in self.buffers:
            if buffer.ring_depth < 1:
                raise PlanVerificationError(f"buffer {buffer.id!r} has invalid ring depth")
            unknown = set(buffer.producer_ids + buffer.consumer_ids) - statement_ids
            if unknown:
                raise PlanVerificationError(
                    f"buffer {buffer.id!r} references unknown statements {sorted(unknown)!r}"
                )
        for hole in self.holes:
            if hole.statement_id not in statement_ids:
                raise PlanVerificationError(
                    f"hole references unknown statement {hole.statement_id!r}"
                )
            if not all(isinstance(item, str) for item in hole.inputs + hole.outputs + hole.relations):
                raise PlanVerificationError(f"hole {hole.statement_id!r} has malformed relations")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    def render(self) -> str:
        lines = ["pipeline schedule"]
        lines.extend(
            f"{item.id}: {item.instruction} [{item.start}, {item.end})"
            for item in self.statements
        )
        return "\n".join(lines)


def export_pipeline_plan(
    program: PipelineProgram, solution: PipelineSolution, target: object
) -> PipelineSchedulePlan:
    """Export stable values only; no target or solver value escapes the plan."""
    from tilefoundry.schedule.render import emit_scaffold  # noqa: PLC0415

    by_id = {item.id: item for item in solution.statements}
    if tuple(by_id) != tuple(unit.name for unit in program.units):
        raise PlanVerificationError("solution statements do not match the pipeline program")
    ring = {item.id: item.ring_depth for item in solution.buffers}
    skeleton, _swimlane, contracts = emit_scaffold(program.graph, program.tree, ring)
    statements = tuple(
        ScheduledStatement(
            id=unit.name,
            instruction=by_id[unit.name].instruction.atom.op.name,
            tile=tuple(
                int(domain.dim_max_val(axis).num_si()) - int(domain.dim_min_val(axis).num_si()) + 1
                for domain in _domains(program, unit.name)
                for axis in range(domain.dim(1))
            ),
            resources=by_id[unit.name].resources,
            start=by_id[unit.name].start,
            end=by_id[unit.name].end,
        )
        for unit in program.units
    )
    buffers = tuple(
        ScheduledBuffer(
            id=item.id,
            storage="smem",
            ring_depth=item.ring_depth,
            producer_ids=item.producer_ids,
            consumer_ids=item.consumer_ids,
        )
        for item in solution.buffers
    )
    holes = tuple(
        KernelHole(
            statement_id=contract.name.removeprefix("HOLE_"),
            inputs=tuple(item.tensor_name for item in contract.inputs),
            outputs=(contract.output.tensor_name,),
            relations=tuple(str(item.index_map) for item in contract.inputs)
            + (str(contract.output.index_map),),
        )
        for contract in contracts
    )
    return PipelineSchedulePlan(
        target=TargetSpecRef(
            architecture_id=getattr(target, "architecture_id", None)
            or target.architecture.name,
            architecture_digest=getattr(target, "architecture_digest", None) or "",
            device_id=getattr(target, "device_id", None) or target.device.name,
            device_digest=getattr(target, "device_digest", None) or "",
        ),
        scaffold=skeleton.text,
        statements=statements,
        buffers=buffers,
        holes=holes,
    )


def _domains(program: PipelineProgram, name: str) -> tuple[object, ...]:
    """Return the one ISL domain piece named by a stable statement ID."""
    domains: list[object] = []
    program.graph.domain.foreach_set(
        lambda domain: domains.append(domain) if domain.get_tuple_name() == name else None
    )
    if len(domains) != 1:
        raise PlanVerificationError(f"statement {name!r} has {len(domains)} domain pieces")
    return tuple(domains)


__all__ = [
    "KernelHole",
    "PipelineSchedulePlan",
    "ScheduledBuffer",
    "ScheduledStatement",
    "TargetSpecRef",
    "export_pipeline_plan",
]
