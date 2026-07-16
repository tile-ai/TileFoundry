from __future__ import annotations

import math
from dataclasses import dataclass

from .space import EdgeKind, PlacementOption


@dataclass(frozen=True, slots=True)
class NodeAssignment:
    node: int
    option: int
    placement: PlacementOption
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_time) or not math.isfinite(self.end_time):
            raise ValueError("node assignment times must be finite")
        if self.start_time < 0 or self.end_time < self.start_time:
            raise ValueError("node assignment times must be ordered and non-negative")

    @property
    def axis_starts(self) -> tuple[int, ...]:
        return self.placement.axis_starts

    @property
    def axis_extents(self) -> tuple[int, ...]:
        return self.placement.axis_extents


@dataclass(frozen=True, slots=True)
class EdgeAssignment:
    use: int
    option: int
    kind: EdgeKind
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_time) or not math.isfinite(self.end_time):
            raise ValueError("edge assignment times must be finite")
        if self.start_time < 0 or self.end_time < self.start_time:
            raise ValueError("edge assignment times must be ordered and non-negative")


@dataclass(frozen=True, slots=True)
class ScheduleSolution:
    node_assignments: tuple[NodeAssignment, ...]
    edge_assignments: tuple[EdgeAssignment, ...]
    makespan: float
    problem_fingerprint: str

    def __post_init__(self) -> None:
        if not self.problem_fingerprint:
            raise ValueError("ScheduleSolution requires a problem fingerprint")
        if not math.isfinite(self.makespan) or self.makespan < 0:
            raise ValueError("ScheduleSolution makespan must be finite and non-negative")
        if len({item.node for item in self.node_assignments}) != len(self.node_assignments):
            raise ValueError("ScheduleSolution has duplicate node assignments")
        if len({item.use for item in self.edge_assignments}) != len(self.edge_assignments):
            raise ValueError("ScheduleSolution has duplicate edge assignments")

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


__all__ = [
    "EdgeAssignment",
    "NodeAssignment",
    "ScheduleSolution",
]
