"""Represent a solved spatial partition without rewriting its program.

Values and operations use authored identities and retain their IR types.
Verification resolves references, checks placement edges and level bounds, and
rejects overlapping operations without rebuilding candidates or re-solving.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

from tilefoundry.ir.core.metadata import binding_name
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.function import Function
from tilefoundry.ir.types import TensorType, Type
from tilefoundry.ir.types.shape_helpers import static_dim_value
from tilefoundry.ir.types.shard import ShardLayout, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.schedule.plan import (
    PlanVerificationError,
    SchedulePlan,
    TargetSpecRef,
)

from .problem import PartitionProblem
from .solve import PartitionSolution


@dataclass(frozen=True)
class PositionInterval:
    """The half-open range of parallel positions something occupies."""

    start: int
    end: int


@dataclass(frozen=True)
class TimeInterval:
    """One operation's half-open execution interval."""

    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class PlacedValue:
    """One tensor value, the type it was placed in, and who touches it."""

    id: str
    type: Type
    producer_id: str | None
    consumer_ids: tuple[str, ...]
    positions: PositionInterval


@dataclass(frozen=True)
class PartitionedOperation:
    """One operation that runs, where it runs, and when.

    `synthesized` marks a Reshard the algorithm introduced to connect two
    otherwise unconnected placements. It is one of these records like any other
    operation, so there is no second channel an agent would have to read to learn
    that data moves.

    `positions` is absent for an operation that occupies no parallel position of
    its own. Moving a value between placements is charged as traffic rather than
    as occupancy, so a Reshard has no position range to state.
    """

    id: str
    operation: str
    synthesized: bool
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    positions: PositionInterval | None
    interval: TimeInterval | None


@dataclass(frozen=True)
class PartitionProof:
    """What the solve proved about its own objective.

    This is a result fact, not a summary of the run: the objective value, the
    bound the solver could establish, and whether the two met.
    """

    status: Literal["OPTIMAL", "FEASIBLE_NOT_PROVEN"]
    objective_ns: int
    best_bound_ns: int
    proven_optimal: bool


def _logical_tensor(type: Type) -> tuple[object, ...] | None:
    """What stays the same across every placement of one tensor."""
    if not isinstance(type, TensorType):
        return None
    return (type.shape, type.dtype, type.storage)


def _layout_json(type: Type) -> object:
    """The placement part of one type, as plain data."""
    if not isinstance(type, TensorType) or not isinstance(type.layout, ShardLayout):
        return None
    layout = type.layout
    return {
        "topology": layout.mesh.topologies[0].name,
        "mesh_shape": [str(dim) for dim in layout.mesh.layout.shape],
        "attrs": [attr.__class__.__name__ for attr in layout.attrs],
        "shape": [str(dim) for dim in layout.layout.shape],
        "strides": (
            None
            if layout.layout.strides is None
            else [str(stride) for stride in layout.layout.strides]
        ),
    }


def _type_json(type: Type) -> object:
    """One selected type as plain data, in the plan's own vocabulary."""
    if not isinstance(type, TensorType):
        return {"kind": type.__class__.__name__}
    return {
        "kind": "tensor",
        "shape": [str(dim) for dim in type.shape],
        "dtype": type.dtype.name,
        "storage": type.storage.name.lower(),
        "layout": _layout_json(type),
    }


@dataclass(frozen=True)
class PartitionSchedulePlan(SchedulePlan):
    """The placement one partition solve committed to, and its proof."""

    topology: str
    extent: int
    target: TargetSpecRef
    values: tuple[PlacedValue, ...]
    operations: tuple[PartitionedOperation, ...]
    root_results: tuple[str, ...]
    proof: PartitionProof

    def verify(self, module: Module, function: Function, topology: Topology) -> None:
        """Check the decision holds together, without invoking a solver."""
        self._check_request(topology)
        values = self._checked_index()
        self._check_references(values)
        self._check_edges(values)
        self._check_positions(values)
        self._check_exclusion()
        self._check_roots(values)
        if self.proof.best_bound_ns > self.proof.objective_ns:
            raise PlanVerificationError(
                "partition plan states a bound above its own objective"
            )

    def _check_request(self, topology: Topology) -> None:
        if topology.name != self.topology:
            raise PlanVerificationError(
                f"partition plan decided topology {self.topology!r}, not "
                f"{topology.name!r}"
            )
        extent = static_dim_value(topology.size)
        if extent != self.extent:
            raise PlanVerificationError(
                f"partition plan decided over {self.extent} positions of "
                f"{self.topology!r}, but the level declares {extent}"
            )

    def _checked_index(self) -> dict[str, PlacedValue]:
        values: dict[str, PlacedValue] = {}
        for value in self.values:
            if value.id in values:
                raise PlanVerificationError(
                    f"partition plan places value {value.id!r} twice"
                )
            values[value.id] = value
        seen: set[str] = set()
        for operation in self.operations:
            if operation.id in seen:
                raise PlanVerificationError(
                    f"partition plan runs operation {operation.id!r} twice"
                )
            seen.add(operation.id)
        return values

    def _check_references(self, values: dict[str, PlacedValue]) -> None:
        """Every edge must be named the same way from both of its ends.

        Naming an operation that exists is not enough: an edge that one end
        claims and the other does not is a claim about a decision nobody made,
        and a reachability walk that followed it would report a program flow that
        the operations do not implement.
        """
        operations = {operation.id: operation for operation in self.operations}
        for value in self.values:
            producer = value.producer_id
            if producer is not None:
                if producer not in operations:
                    raise PlanVerificationError(
                        f"partition plan value {value.id!r} names producer "
                        f"{producer!r}, which the plan does not run"
                    )
                if value.id not in operations[producer].output_ids:
                    raise PlanVerificationError(
                        f"partition plan value {value.id!r} names producer "
                        f"{producer!r}, which does not produce it"
                    )
            for consumer_id in value.consumer_ids:
                if consumer_id not in operations:
                    raise PlanVerificationError(
                        f"partition plan value {value.id!r} names consumer "
                        f"{consumer_id!r}, which the plan does not run"
                    )
                if value.id not in operations[consumer_id].input_ids:
                    raise PlanVerificationError(
                        f"partition plan value {value.id!r} names consumer "
                        f"{consumer_id!r}, which does not read it"
                    )
        for operation in self.operations:
            for value_id in (*operation.input_ids, *operation.output_ids):
                if value_id not in values:
                    raise PlanVerificationError(
                        f"partition plan operation {operation.id!r} refers to "
                        f"unplaced value {value_id!r}"
                    )
            for value_id in operation.output_ids:
                if values[value_id].producer_id != operation.id:
                    raise PlanVerificationError(
                        f"partition plan operation {operation.id!r} produces "
                        f"{value_id!r}, which names producer "
                        f"{values[value_id].producer_id!r}"
                    )
            for value_id in operation.input_ids:
                if operation.id not in values[value_id].consumer_ids:
                    raise PlanVerificationError(
                        f"partition plan operation {operation.id!r} reads "
                        f"{value_id!r}, which does not name it as a consumer"
                    )

    def _check_edges(self, values: dict[str, PlacedValue]) -> None:
        for value in self.values:
            if not isinstance(value.type, TensorType):
                raise PlanVerificationError(
                    f"partition plan places value {value.id!r} in "
                    f"{type(value.type).__name__}, which is not a tensor type"
                )
            if value.type.storage is not StorageKind.GMEM:
                raise PlanVerificationError(
                    f"partition plan places value {value.id!r} in "
                    f"{value.type.storage.name}, and a partitioned value is "
                    "addressable global memory"
                )
        for operation in self.operations:
            if not operation.synthesized:
                continue
            inputs = tuple(values[value_id] for value_id in operation.input_ids)
            outputs = tuple(values[value_id] for value_id in operation.output_ids)
            if len(inputs) != 1 or len(outputs) != 1:
                raise PlanVerificationError(
                    f"partition plan synthesized {operation.id!r} with "
                    f"{len(inputs)} inputs and {len(outputs)} outputs; moving one "
                    "value takes one of each"
                )
            source, target = inputs[0], outputs[0]
            if _logical_tensor(source.type) != _logical_tensor(target.type):
                raise PlanVerificationError(
                    f"partition plan synthesized {operation.id!r} between "
                    f"{source.id!r} and {target.id!r}, which are different logical "
                    "tensors"
                )
            if source.type == target.type:
                raise PlanVerificationError(
                    f"partition plan synthesized {operation.id!r} between two "
                    "identical placements, which moves nothing"
                )

    def _check_positions(self, values: dict[str, PlacedValue]) -> None:
        for value in self.values:
            self._check_interval(value.id, value.positions)
        for operation in self.operations:
            if operation.positions is not None:
                self._check_interval(operation.id, operation.positions)
            if operation.interval is not None and (
                operation.interval.end_ns < operation.interval.start_ns
            ):
                raise PlanVerificationError(
                    f"partition plan operation {operation.id!r} ends before it starts"
                )

    def _check_interval(self, owner: str, positions: PositionInterval) -> None:
        if positions.end <= positions.start:
            raise PlanVerificationError(
                f"partition plan gives {owner!r} the empty position range "
                f"[{positions.start}, {positions.end})"
            )
        if positions.start < 0 or positions.end > self.extent:
            raise PlanVerificationError(
                f"partition plan places {owner!r} on positions [{positions.start}, "
                f"{positions.end}), outside the {self.extent} positions of "
                f"{self.topology!r}"
            )

    def _check_exclusion(self) -> None:
        """No two operations may hold the same position at the same time.

        Only operations that occupy positions and run over an interval take part:
        an operation charged as traffic states no occupancy to conflict over.
        """
        placed = tuple(
            operation
            for operation in self.operations
            if operation.interval is not None and operation.positions is not None
        )
        for index, left in enumerate(placed):
            for right in placed[index + 1 :]:
                if not _overlap(
                    left.interval.start_ns,
                    left.interval.end_ns,
                    right.interval.start_ns,
                    right.interval.end_ns,
                ):
                    continue
                if _overlap(
                    left.positions.start,
                    left.positions.end,
                    right.positions.start,
                    right.positions.end,
                ):
                    raise PlanVerificationError(
                        f"partition plan runs {left.id!r} and {right.id!r} at the "
                        "same time on overlapping positions"
                    )

    def _check_roots(self, values: dict[str, PlacedValue]) -> None:
        """Every root result must be reachable by following producer edges.

        A placement the plan does not produce is where the walk stops: it is
        either the program's own input or a value a region carries, and neither is
        an operation this plan decided about.
        """
        available = {value.id for value in self.values if value.producer_id is None}
        producer_of = {
            value.id: value.producer_id
            for value in self.values
            if value.producer_id is not None
        }
        inputs_of = {operation.id: operation.input_ids for operation in self.operations}

        def reachable(value_id: str, active: frozenset[str]) -> bool:
            if value_id in available:
                return True
            if value_id in active:
                return False
            producer = producer_of.get(value_id)
            if producer is None:
                return False
            return all(
                reachable(source, active | {value_id}) for source in inputs_of[producer]
            )

        for value_id in self.root_results:
            if value_id not in values:
                raise PlanVerificationError(
                    f"partition plan leaves root result {value_id!r} unplaced"
                )
            if not reachable(value_id, frozenset()):
                raise PlanVerificationError(
                    f"partition plan cannot reach root result {value_id!r} from the "
                    "program's own inputs"
                )

    def to_json(self) -> str:
        """Render the whole decision as sorted-key JSON."""
        payload = {
            "topology": self.topology,
            "extent": self.extent,
            "target": asdict(self.target),
            "proof": asdict(self.proof),
            "root_results": list(self.root_results),
            "values": [
                {
                    "id": value.id,
                    "type": _type_json(value.type),
                    "producer_id": value.producer_id,
                    "consumer_ids": list(value.consumer_ids),
                    "positions": asdict(value.positions),
                }
                for value in self.values
            ],
            "operations": [
                {
                    "id": operation.id,
                    "operation": operation.operation,
                    "synthesized": operation.synthesized,
                    "input_ids": list(operation.input_ids),
                    "output_ids": list(operation.output_ids),
                    "positions": (
                        asdict(operation.positions)
                        if operation.positions is not None
                        else None
                    ),
                    "interval": (
                        asdict(operation.interval)
                        if operation.interval is not None
                        else None
                    ),
                }
                for operation in self.operations
            ],
        }
        return json.dumps(payload, sort_keys=True)


def _overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _base_value_name(problem: PartitionProblem, value_id: int) -> str:
    """A readable name for one value, derived from the authored program."""
    info = problem.values[value_id]
    name = binding_name(info.source) or getattr(info.source, "name", None)
    if not name:
        target = getattr(info.source, "target", None)
        name = type(target).__name__.lower() if target is not None else "value"
    for index in info.leaf_path:
        name = f"{name}.{index}"
    if info.role != "normal":
        name = f"{name}.{info.role}"
    return name


def _placement_tag(type: Type) -> str:
    """A short readable name for how one placement divides its value."""
    if not isinstance(type, TensorType) or not isinstance(type.layout, ShardLayout):
        return "whole"
    kinds = {"Split": "split", "Broadcast": "bcast", "Partial": "partial"}
    attrs = "".join(
        kinds.get(attr.__class__.__name__, attr.__class__.__name__.lower())
        for attr in type.layout.attrs
    )
    extent = type.layout.mesh.layout.shape[0]
    return f"{attrs or 'whole'}{extent}"


def _placement_ids(
    problem: PartitionProblem, selected_buckets: tuple[int, ...]
) -> dict[int, str]:
    """One stable readable identity per selected placement.

    A value may be resident in more than one placement at once: that is exactly
    what a Reshard connects. The value's own name is used alone while it has a
    single placement, and is qualified by how each placement divides it when it
    has several, so a plan naming two placements of one tensor stays readable.
    """
    by_value: dict[int, list[int]] = {}
    for bucket_id in sorted(selected_buckets):
        by_value.setdefault(problem.buckets[bucket_id].value_id, []).append(bucket_id)
    used: dict[str, int] = {}
    ids: dict[int, str] = {}
    for value_id in sorted(by_value):
        buckets = by_value[value_id]
        name = _base_value_name(problem, value_id)
        for bucket_id in buckets:
            base = name
            if len(buckets) > 1:
                type = problem.types[problem.buckets[bucket_id].type_id]
                base = f"{name}@{_placement_tag(type)}"
            count = used.get(base, 0)
            used[base] = count + 1
            ids[bucket_id] = base if count == 0 else f"{base}#{count}"
    return ids


def _operation_ids(
    problem: PartitionProblem,
    placement_ids: dict[int, str],
    selected: tuple[int, ...],
) -> dict[int, str]:
    """One stable readable identity per selected operation."""
    ids: dict[int, str] = {}
    used: dict[str, int] = {}
    for candidate_id in selected:
        candidate = problem.candidates[candidate_id]
        kind = type(candidate.op).__name__.lower()
        produced = tuple(
            placement_ids[bucket_id]
            for bucket_id in candidate.output_bucket_ids
            if bucket_id in placement_ids
        )
        base = f"{kind}:{produced[0]}" if produced else kind
        count = used.get(base, 0)
        used[base] = count + 1
        ids[candidate_id] = base if count == 0 else f"{base}#{count}"
    return ids


def _placed_positions(problem: PartitionProblem, bucket_id: int | None) -> int:
    """How many positions one selected placement occupies."""
    if bucket_id is None:
        return 1
    type = problem.types[problem.buckets[bucket_id].type_id]
    if not isinstance(type, TensorType) or not isinstance(type.layout, ShardLayout):
        return 1
    count = type.layout.mesh.layout.shape[0]
    return count if isinstance(count, int) and count > 0 else 1


def export_partition_plan(
    problem: PartitionProblem, solution: PartitionSolution
) -> PartitionSchedulePlan:
    """State the solved selection in the plan's own stable vocabulary."""
    placement_ids = _placement_ids(problem, solution.selected_bucket_ids)
    operation_ids = _operation_ids(
        problem, placement_ids, solution.selected_candidate_ids
    )
    intervals = dict(solution.candidate_intervals_ns)
    offsets = dict(solution.bucket_offsets)

    producers: dict[str, str] = {}
    consumers: dict[str, list[str]] = {}
    operations: list[PartitionedOperation] = []
    for candidate_id in solution.selected_candidate_ids:
        candidate = problem.candidates[candidate_id]
        operation_id = operation_ids[candidate_id]
        input_ids = tuple(
            placement_ids[bucket_id] for bucket_id in candidate.input_bucket_ids
        )
        output_ids = tuple(
            placement_ids[bucket_id] for bucket_id in candidate.output_bucket_ids
        )
        for placement in output_ids:
            producers[placement] = operation_id
        for placement in input_ids:
            consumers.setdefault(placement, []).append(operation_id)
        anchor = candidate.output_bucket_ids[0] if candidate.output_bucket_ids else None
        interval = intervals.get(candidate_id)
        positions = None
        if candidate.topology_count > 0 and anchor is not None:
            start = offsets.get(anchor, 0)
            positions = PositionInterval(start, start + candidate.topology_count)
        operations.append(
            PartitionedOperation(
                id=operation_id,
                operation=type(candidate.op).__name__,
                synthesized=candidate.site_id is None,
                input_ids=input_ids,
                output_ids=output_ids,
                positions=positions,
                interval=(
                    TimeInterval(interval.start_ns, interval.end_ns)
                    if interval is not None
                    else None
                ),
            )
        )

    values = tuple(
        PlacedValue(
            id=placement_ids[bucket_id],
            type=problem.types[problem.buckets[bucket_id].type_id],
            producer_id=producers.get(placement_ids[bucket_id]),
            consumer_ids=tuple(consumers.get(placement_ids[bucket_id], ())),
            positions=PositionInterval(
                offsets.get(bucket_id, 0),
                offsets.get(bucket_id, 0) + _placed_positions(problem, bucket_id),
            ),
        )
        for bucket_id in sorted(solution.selected_bucket_ids)
    )
    root_placements = tuple(
        placement_ids[bucket_id]
        for bucket_id in sorted(solution.selected_bucket_ids)
        if problem.buckets[bucket_id].value_id in set(problem.root_value_ids)
    )

    return PartitionSchedulePlan(
        topology=problem.topology.name,
        extent=problem.extent,
        target=problem.facts.spec,
        values=values,
        operations=tuple(operations),
        root_results=root_placements,
        proof=PartitionProof(
            status=solution.status,
            objective_ns=solution.makespan_ns,
            best_bound_ns=solution.best_bound_ns,
            proven_optimal=solution.status == "OPTIMAL",
        ),
    )


__all__ = [
    "PartitionProof",
    "PartitionSchedulePlan",
    "PartitionedOperation",
    "PlacedValue",
    "PositionInterval",
    "TimeInterval",
    "export_partition_plan",
]
