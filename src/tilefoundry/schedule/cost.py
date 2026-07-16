from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .space import EdgeOption, NodeOption, ScheduleSpace


@dataclass(frozen=True, slots=True)
class CostEstimate:
    duration: float
    bytes: int = 0
    flops: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration) or self.duration < 0:
            raise ValueError("cost duration must be finite and non-negative")
        if self.bytes < 0 or self.flops < 0:
            raise ValueError("cost work values must be non-negative")


class CostModel(Protocol):
    def estimate_node(self, option: NodeOption, context) -> CostEstimate:
        ...

    def estimate_edge(self, option: EdgeOption, context) -> CostEstimate:
        ...


@dataclass(frozen=True, slots=True)
class CostTable:
    node_costs: tuple[tuple[int, CostEstimate], ...]
    edge_costs: tuple[tuple[int, CostEstimate], ...]

    def __post_init__(self) -> None:
        if len({option_id for option_id, _ in self.node_costs}) != len(self.node_costs):
            raise ValueError("CostTable node option IDs must be unique")
        if len({option_id for option_id, _ in self.edge_costs}) != len(self.edge_costs):
            raise ValueError("CostTable edge option IDs must be unique")

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
    node_costs = tuple(
        (option.id, model.estimate_node(option, context))
        for option in space.node_options
    )
    edge_costs = tuple(
        (option.id, model.estimate_edge(option, context))
        for option in space.edge_options
    )
    return CostTable(node_costs=node_costs, edge_costs=edge_costs)


__all__ = ["CostEstimate", "CostModel", "CostTable", "build_cost_table"]
