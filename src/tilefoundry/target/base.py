"""Immutable compilation target values and exact stage-service lookup."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Target:
    """Identify a compilation backend and its private stage services."""

    name: str
    _services: tuple[tuple[type, str, object], ...] = field(
        default=(), init=False, compare=False, hash=False, repr=False
    )

    def service(self, interface: type, stage: str) -> object:
        """Return the exact service bound to ``(interface, stage)``."""
        if not isinstance(interface, type):
            raise TypeError(
                f"{type(self).__name__}.service: interface must be a type, "
                f"got {type(interface).__name__}"
            )
        if not isinstance(stage, str) or not stage:
            raise ValueError(
                f"{type(self).__name__}.service: stage must be a non-empty string, "
                f"got {stage!r}"
            )
        matches = [
            service
            for bound_interface, bound_stage, service in self._services
            if bound_interface is interface and bound_stage == stage
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{self!r}: expected exactly one service for interface "
                f"{interface.__name__!r} at stage {stage!r}, found {len(matches)}"
            )
        return matches[0]


def bind_services(target: Target, bindings: tuple[tuple[type, str, object], ...]) -> None:
    """Bind an immutable service table during target construction."""
    seen: set[tuple[type, str]] = set()
    for interface, stage, _service in bindings:
        if not isinstance(interface, type):
            raise TypeError("Target service interface must be a type")
        if not isinstance(stage, str) or not stage:
            raise ValueError("Target service stage must be a non-empty string")
        key = (interface, stage)
        if key in seen:
            raise ValueError(
                f"Target: duplicate service binding for "
                f"({interface.__name__}, {stage!r})"
            )
        seen.add(key)
    object.__setattr__(target, "_services", tuple(bindings))


@dataclass(frozen=True)
class CpuTarget(Target):
    """Identify the CPU host backend."""

    name: str = field(default="cpu", init=False)


__all__ = ["CpuTarget", "Target", "bind_services"]
