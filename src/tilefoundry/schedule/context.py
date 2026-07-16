from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class ScheduleContext:
    target: object
    level: str
    mesh: object
    services: object | None = None

    def with_services(self, services: object) -> "ScheduleContext":
        return replace(self, services=services)


__all__ = ["ScheduleContext"]
