from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ServiceCollection:
    """Immutable view of provider services for one target and schedule level."""

    services: tuple[tuple[type, Any], ...]

    @classmethod
    def from_values(cls, **values: Any) -> "ServiceCollection":
        return cls(tuple((type(value), value) for value in values.values()))

    def get(self, service_type: type) -> Any:
        for registered_type, value in self.services:
            if registered_type is service_type or isinstance(value, service_type):
                return value
        raise KeyError(service_type)


@dataclass(frozen=True, slots=True)
class TargetScheduleProfile:
    level: str
    topology: str = "cta"
    topology_rank: int = 1
    max_ctas: int = 132


__all__ = ["ServiceCollection", "TargetScheduleProfile"]
