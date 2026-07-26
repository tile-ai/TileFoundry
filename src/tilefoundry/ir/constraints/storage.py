"""Storage hard constraints."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types.storage import StorageKind, resolve_storage

from .base import ScheduleConstraint


@dataclass(frozen=True)
class StorageConstraint(ScheduleConstraint):
    """Filter a value by one current IR StorageKind."""

    storage: StorageKind | None = None

    def __post_init__(self) -> None:
        value = resolve_storage(self.storage)
        if value is None:
            raise ValueError("storage constraint requires a concrete StorageKind")
        object.__setattr__(self, "storage", value)


__all__ = ["StorageConstraint"]
