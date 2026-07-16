from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tilefoundry.ir.types.shard.mesh import Mesh


@dataclass(frozen=True, slots=True)
class ScheduleContext:
    target: object
    level: str
    mesh: Mesh
    bandwidth: float | None = None
    peak_flops: float | None = None


class TargetScheduleBackend(Protocol):
    def build_space(self, graph, context) -> object:
        ...

    def cost_model(self, context) -> object:
        ...

    def solver(self, context) -> object:
        ...

    def materialize(self, problem, solution, context) -> object:
        ...


_BACKENDS: dict[tuple[type, str], object] = {}


def register_schedule_backend(target_type: type, *, level: str, backend: object) -> None:
    if not isinstance(target_type, type):
        raise TypeError("target_type must be a concrete target type")
    if not level:
        raise ValueError("schedule backend level must not be empty")
    _BACKENDS[(target_type, level)] = backend


def resolve_schedule_backend(target: object, *, level: str):
    backend = _BACKENDS.get((type(target), level))
    if backend is None:
        for (target_type, registered_level), candidate in _BACKENDS.items():
            if registered_level == level and isinstance(target, target_type):
                backend = candidate
                break
    if backend is None:
        raise ValueError(
            f"no schedule backend registered for target {type(target).__name__!r} "
            f"at level {level!r}"
        )
    return backend


__all__ = [
    "ScheduleContext",
    "TargetScheduleBackend",
    "register_schedule_backend",
    "resolve_schedule_backend",
]
