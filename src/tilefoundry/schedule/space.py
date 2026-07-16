from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .candidate import CandidateTable, OpCandidate, Submesh
from .graph import ProgramScheduleGraph


@dataclass(frozen=True, slots=True)
class PhysicalRepresentation:
    id: int
    storage: object
    layout_key: str


@dataclass(frozen=True, slots=True)
class PlacementOption:
    id: int
    parent_mesh: object
    axis_starts: tuple[int, ...]
    axis_extents: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.axis_starts) != len(self.axis_extents) or not self.axis_starts:
            raise ValueError("placement starts and extents must have equal nonzero rank")
        if any(start < 0 or extent <= 0 for start, extent in zip(self.axis_starts, self.axis_extents)):
            raise ValueError("placement starts must be non-negative and extents positive")

    @property
    def submesh(self) -> Submesh:
        return Submesh(self.axis_starts, self.axis_extents)


@dataclass(frozen=True, slots=True)
class NodeOption:
    id: int
    node: int
    candidate: OpCandidate
    implementation_key: str
    placements: tuple[PlacementOption, ...]
    input_representations: tuple[int, ...]
    output_representations: tuple[int, ...]
    lowering_key: str

    @property
    def placement(self) -> PlacementOption:
        if len(self.placements) != 1:
            raise ValueError("NodeOption has multiple placement choices")
        return self.placements[0]


class EdgeKind(str, Enum):
    DIRECT = "direct"
    RESHARD = "reshard"


@dataclass(frozen=True, slots=True)
class EdgeOption:
    id: int
    use: int
    kind: EdgeKind
    source_representation: int
    destination_representation: int
    same_placement_required: bool
    payload_bytes: int
    cta_count: int


@dataclass(frozen=True, slots=True)
class Resource:
    id: int
    name: str
    capacity: int


@dataclass(frozen=True, slots=True)
class ScheduleSpace:
    node_options: tuple[NodeOption, ...]
    edge_options: tuple[EdgeOption, ...]
    resources: tuple[Resource, ...]
    representations: tuple[PhysicalRepresentation, ...] = ()
    candidates: CandidateTable | None = None

    def __post_init__(self) -> None:
        for values, label in (
            (self.node_options, "node option"),
            (self.edge_options, "edge option"),
            (self.resources, "resource"),
            (self.representations, "representation"),
        ):
            ids = [value.id for value in values]
            if len(set(ids)) != len(ids):
                raise ValueError(f"ScheduleSpace {label} IDs must be unique")

    def options_for_node(self, node_id: int) -> tuple[NodeOption, ...]:
        return tuple(option for option in self.node_options if option.node == node_id)

    def options_for_use(self, use_id: int) -> tuple[EdgeOption, ...]:
        return tuple(option for option in self.edge_options if option.use == use_id)

    def representation(self, representation_id: int) -> PhysicalRepresentation:
        for representation in self.representations:
            if representation.id == representation_id:
                return representation
        raise KeyError(representation_id)


def _parent_extent(mesh: object) -> int:
    shape = tuple(getattr(mesh, "shape", ()))
    if len(shape) != 1 or not isinstance(shape[0], int) or shape[0] <= 0:
        raise ValueError("CTA AutoDist v1 requires a static one-dimensional parent mesh")
    return shape[0]


def _placement_starts(parent_extent: int, extent: int) -> tuple[int, ...]:
    if extent > parent_extent:
        return ()
    if parent_extent <= 32:
        return tuple(range(parent_extent - extent + 1))
    starts = {0, parent_extent - extent, (parent_extent - extent) // 2}
    return tuple(sorted(starts))


def _value_bytes(value: object) -> int:
    ty = getattr(value, "type", None)
    shape = getattr(ty, "shape", ())
    if not shape or not all(isinstance(dim, int) for dim in shape):
        return 4
    itemsize = {
        "f32": 4,
        "f16": 2,
        "bf16": 2,
        "fp8e4m3": 1,
        "f8e8m0": 1,
        "f4e2m1": 1,
        "i32": 4,
        "i64": 8,
        "bool": 1,
    }.get(getattr(getattr(ty, "dtype", None), "value", "f32"), 4)
    return max(1, math.prod(shape) * itemsize)


def build_schedule_space(
    graph: ProgramScheduleGraph,
    candidates: CandidateTable,
    *,
    parent_mesh: object,
) -> ScheduleSpace:
    """Lift common candidates into finite placement and edge choices."""
    parent_extent = _parent_extent(parent_mesh)
    representations = (PhysicalRepresentation(0, "gmem", "logical"),)
    node_options: list[NodeOption] = []
    placement_id = 0
    option_id = 0
    for op in graph.ops:
        for candidate in candidates.for_op(op.id):
            placements: list[PlacementOption] = []
            for extent in (candidate.cta_count,):
                for start in _placement_starts(parent_extent, extent):
                    placements.append(
                        PlacementOption(
                            id=placement_id,
                            parent_mesh=parent_mesh,
                            axis_starts=(start,),
                            axis_extents=(extent,),
                        )
                    )
                    placement_id += 1
            if not placements:
                continue
            node_options.append(
                NodeOption(
                    id=option_id,
                    node=op.id,
                    candidate=candidate,
                    implementation_key=candidate.implementation_key,
                    placements=tuple(placements),
                    input_representations=tuple(0 for _ in op.inputs),
                    output_representations=(0,),
                    lowering_key="ordinary_hir_op",
                )
            )
            option_id += 1

    edge_options: list[EdgeOption] = []
    edge_id = 0
    for edge in graph.edges:
        payload = _value_bytes(graph.value(edge.source).ir_value)
        edge_options.append(
            EdgeOption(
                id=edge_id,
                use=edge.id,
                kind=EdgeKind.DIRECT,
                source_representation=0,
                destination_representation=0,
                same_placement_required=edge.kind in {"data", "call_result"},
                payload_bytes=payload,
                cta_count=parent_extent,
            )
        )
        edge_id += 1
        edge_options.append(
            EdgeOption(
                id=edge_id,
                use=edge.id,
                kind=EdgeKind.RESHARD,
                source_representation=0,
                destination_representation=0,
                same_placement_required=False,
                payload_bytes=payload,
                cta_count=parent_extent,
            )
        )
        edge_id += 1

    return ScheduleSpace(
        node_options=tuple(node_options),
        edge_options=tuple(edge_options),
        resources=(Resource(0, "cta", parent_extent),),
        representations=representations,
        candidates=candidates,
    )


__all__ = [
    "EdgeKind",
    "EdgeOption",
    "NodeOption",
    "PhysicalRepresentation",
    "PlacementOption",
    "Resource",
    "ScheduleSpace",
    "build_schedule_space",
]
