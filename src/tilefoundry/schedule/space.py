from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tilefoundry.ir.target.storage import StorageKind
from tilefoundry.ir.types.shard.mesh import Mesh


@dataclass(frozen=True, slots=True)
class PhysicalRepresentation:
    id: int
    storage: StorageKind
    layout_key: str


@dataclass(frozen=True, slots=True)
class PlacementOption:
    id: int
    parent_mesh: Mesh
    axis_starts: tuple[int, ...]
    axis_extents: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.axis_starts or len(self.axis_starts) != len(self.axis_extents):
            raise ValueError("placement starts and extents must have equal nonzero rank")
        if any(type(value) is not int or value < 0 for value in self.axis_starts):
            raise ValueError("placement starts must be non-negative integers")
        if any(type(value) is not int or value <= 0 for value in self.axis_extents):
            raise ValueError("placement extents must be positive integers")


@dataclass(frozen=True, slots=True)
class NodeOption:
    id: int
    node: int
    implementation_key: str
    placements: tuple[PlacementOption, ...]
    input_representations: tuple[int, ...]
    output_representations: tuple[int, ...]
    lowering_key: str
    work_bytes: int
    flops: int

    def __post_init__(self) -> None:
        if not self.placements:
            raise ValueError("NodeOption must have at least one placement")
        if self.work_bytes < 0 or self.flops < 0:
            raise ValueError("NodeOption work values must be non-negative")

    @property
    def placement(self) -> PlacementOption:
        if len(self.placements) != 1:
            raise ValueError("MVP node options expose one selected placement")
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

    def __post_init__(self) -> None:
        if self.payload_bytes < 0:
            raise ValueError("EdgeOption payload_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class Resource:
    id: int
    name: str
    capacity: int

    def __post_init__(self) -> None:
        if not self.name or self.capacity <= 0:
            raise ValueError("resources require a name and positive capacity")


@dataclass(frozen=True, slots=True)
class ScheduleSpace:
    node_options: tuple[NodeOption, ...]
    edge_options: tuple[EdgeOption, ...]
    resources: tuple[Resource, ...]
    representations: tuple[PhysicalRepresentation, ...] = ()

    def __post_init__(self) -> None:
        if len({option.id for option in self.node_options}) != len(self.node_options):
            raise ValueError("ScheduleSpace node option IDs must be unique")
        if len({option.id for option in self.edge_options}) != len(self.edge_options):
            raise ValueError("ScheduleSpace edge option IDs must be unique")
        if len({resource.id for resource in self.resources}) != len(self.resources):
            raise ValueError("ScheduleSpace resource IDs must be unique")
        if len({rep.id for rep in self.representations}) != len(self.representations):
            raise ValueError("ScheduleSpace representation IDs must be unique")

    def options_for_node(self, node_id: int) -> tuple[NodeOption, ...]:
        return tuple(option for option in self.node_options if option.node == node_id)

    def options_for_use(self, use_id: int) -> tuple[EdgeOption, ...]:
        return tuple(option for option in self.edge_options if option.use == use_id)

    def representation(self, representation_id: int) -> PhysicalRepresentation:
        for representation in self.representations:
            if representation.id == representation_id:
                return representation
        raise KeyError(representation_id)


__all__ = [
    "EdgeKind",
    "EdgeOption",
    "NodeOption",
    "PhysicalRepresentation",
    "PlacementOption",
    "Resource",
    "ScheduleSpace",
]
