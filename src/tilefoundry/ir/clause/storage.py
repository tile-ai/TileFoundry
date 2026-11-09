"""Storage hard constraints."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr

from .base import Where


@dataclass(frozen=True)
class Storage(Where):
    """Filter a value by one current IR StorageKind."""

    storage: expr.StorageKind | None = None

    def __post_init__(self) -> None:
        value = expr.resolve_storage(self.storage)
        if value is None:
            raise ValueError("storage constraint requires a concrete StorageKind")
        object.__setattr__(self, "storage", value)


__all__ = ["Storage"]
