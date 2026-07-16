from __future__ import annotations

from dataclasses import dataclass

from .candidate import Submesh
from .space import EdgeKind


@dataclass(frozen=True, slots=True)
class OpPlacement:
    start_ns: int
    end_ns: int
    submesh: Submesh

    def __post_init__(self) -> None:
        if self.start_ns < 0 or self.end_ns < self.start_ns:
            raise ValueError("operation placement times must be ordered and non-negative")


@dataclass(frozen=True, slots=True)
class NodeAssignment:
    node: int
    candidate: int
    option: int
    placement: OpPlacement

    @property
    def start_ns(self) -> int:
        return self.placement.start_ns

    @property
    def end_ns(self) -> int:
        return self.placement.end_ns

    @property
    def start_time(self) -> int:
        return self.start_ns

    @property
    def end_time(self) -> int:
        return self.end_ns

    @property
    def axis_starts(self) -> tuple[int, ...]:
        return self.placement.submesh.offsets

    @property
    def axis_extents(self) -> tuple[int, ...]:
        return self.placement.submesh.extents


@dataclass(frozen=True, slots=True)
class EdgeAssignment:
    use: int
    option: int
    kind: EdgeKind
    start_ns: int
    end_ns: int

    @property
    def start_time(self) -> int:
        return self.start_ns

    @property
    def end_time(self) -> int:
        return self.end_ns


@dataclass(frozen=True, slots=True)
class ScheduleSolution:
    node_assignments: tuple[NodeAssignment, ...]
    edge_assignments: tuple[EdgeAssignment, ...]
    makespan_ns: int
    problem_fingerprint: str
    status: str = "OPTIMAL"

    @property
    def makespan(self) -> int:
        return self.makespan_ns

    @property
    def assignments(self) -> tuple[NodeAssignment, ...]:
        return self.node_assignments

    @property
    def edges(self) -> tuple[EdgeAssignment, ...]:
        return self.edge_assignments

    def assignment_for(self, node_id: int) -> NodeAssignment:
        for assignment in self.node_assignments:
            if assignment.node == node_id:
                return assignment
        raise KeyError(node_id)

    def edge_for(self, use_id: int) -> EdgeAssignment:
        for assignment in self.edge_assignments:
            if assignment.use == use_id:
                return assignment
        raise KeyError(use_id)


__all__ = ["EdgeAssignment", "NodeAssignment", "OpPlacement", "ScheduleSolution"]
