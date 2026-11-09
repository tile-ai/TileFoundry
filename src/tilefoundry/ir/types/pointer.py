"""Typed physical-memory pointer descriptors."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr


@dataclass(frozen=True)
class PointerType:
    """A typed physical-memory engine consumed by ``T.tensor_view``."""

    dtype: expr.DType
    storage: expr.StorageKind

    def __post_init__(self) -> None:
        """Normalize the storage."""
        normalized = expr.resolve_storage(self.storage)
        if normalized is None:
            raise TypeError("PointerType.storage must be a StorageKind, not None")
        if normalized is not self.storage:
            object.__setattr__(self, "storage", normalized)


__all__ = ["PointerType"]
