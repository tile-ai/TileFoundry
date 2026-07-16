from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .space import EdgeOption, NodeOption, ScheduleSpace


@dataclass(frozen=True, slots=True)
class CostEstimate:
    duration_ns: int
    traffic_bytes: int = 0
    flops: int = 0
    compute_time_ns: int = 0
    memory_time_ns: int = 0

    def __post_init__(self) -> None:
        if self.duration_ns < 0 or self.traffic_bytes < 0 or self.flops < 0:
            raise ValueError("cost values must be non-negative")
        if not all(math.isfinite(float(value)) for value in (
            self.duration_ns,
            self.traffic_bytes,
            self.flops,
            self.compute_time_ns,
            self.memory_time_ns,
        )):
            raise ValueError("cost values must be finite")

    @property
    def duration(self) -> float:
        return float(self.duration_ns)

    @property
    def bytes(self) -> int:
        return self.traffic_bytes


class CostModel(Protocol):
    def estimate_node(self, option: NodeOption, context) -> CostEstimate:
        ...

    def estimate_edge(self, option: EdgeOption, context) -> CostEstimate:
        ...


@dataclass(frozen=True, slots=True)
class CostTable:
    node_costs: tuple[tuple[int, CostEstimate], ...]
    edge_costs: tuple[tuple[int, CostEstimate], ...]

    def node(self, option_id: int) -> CostEstimate:
        for candidate_id, estimate in self.node_costs:
            if candidate_id == option_id:
                return estimate
        raise KeyError(option_id)

    def edge(self, option_id: int) -> CostEstimate:
        for candidate_id, estimate in self.edge_costs:
            if candidate_id == option_id:
                return estimate
        raise KeyError(option_id)

    def all(self) -> tuple[CostEstimate, ...]:
        return tuple(estimate for _, estimate in (*self.node_costs, *self.edge_costs))


def build_cost_table(space: ScheduleSpace, model: CostModel, context) -> CostTable:
    table = CostTable(
        node_costs=tuple((option.id, model.estimate_node(option, context)) for option in space.node_options),
        edge_costs=tuple((option.id, model.estimate_edge(option, context)) for option in space.edge_options),
    )
    if not all(math.isfinite(float(estimate.duration_ns)) for estimate in table.all()):
        raise ValueError("cost model returned a non-finite estimate")
    return table


__all__ = ["CostEstimate", "CostModel", "CostTable", "build_cost_table"]
