from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.core import Call, Expr
from tilefoundry.ir.hir.function import Function

from .constraints import ConstraintProvenance, SourceLocation, StorageConstraint


@dataclass(frozen=True, slots=True)
class ScheduleRegion:
    id: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    nodes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ScheduleNode:
    id: int
    ir_call: Call
    callee: Function
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ScheduleValue:
    id: int
    ir_value: Expr
    producer: int | None
    consumers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ScheduleUse:
    id: int
    value: int
    consumer: int
    operand_index: int


@dataclass(frozen=True, slots=True)
class GraphStorageConstraint:
    id: int
    target: int
    storage: object
    source_loc: SourceLocation
    provenance: ConstraintProvenance
    authored: StorageConstraint


@dataclass(frozen=True, slots=True)
class ScheduleGraph:
    function: Function
    root: ScheduleRegion
    values: tuple[ScheduleValue, ...]
    uses: tuple[ScheduleUse, ...]
    constraints: tuple[GraphStorageConstraint, ...] = ()
    nodes: tuple[ScheduleNode, ...] = ()

    def node(self, node_id: int) -> ScheduleNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def value(self, value_id: int) -> ScheduleValue:
        for value in self.values:
            if value.id == value_id:
                return value
        raise KeyError(value_id)

    def use(self, use_id: int) -> ScheduleUse:
        for use in self.uses:
            if use.id == use_id:
                return use
        raise KeyError(use_id)

    def producer_of(self, value_id: int) -> ScheduleNode | None:
        producer = self.value(value_id).producer
        return None if producer is None else self.node(producer)

    def __post_init__(self) -> None:
        if len({node.id for node in self.nodes}) != len(self.nodes):
            raise ValueError("ScheduleGraph nodes must have unique IDs")
        if len({value.id for value in self.values}) != len(self.values):
            raise ValueError("ScheduleGraph values must have unique IDs")
        if len({use.id for use in self.uses}) != len(self.uses):
            raise ValueError("ScheduleGraph uses must have unique IDs")
        if len({constraint.id for constraint in self.constraints}) != len(self.constraints):
            raise ValueError("ScheduleGraph constraints must have unique IDs")


__all__ = [
    "GraphStorageConstraint",
    "ScheduleGraph",
    "ScheduleNode",
    "ScheduleRegion",
    "ScheduleUse",
    "ScheduleValue",
]
