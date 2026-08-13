"""Define backend-generic memory-space levels.

Surface strings normalize to ``StorageKind`` at the IR boundary. ``UMAT``
marks compiler-internal values whose residency has not yet been materialized.
``None`` remains a valid optional attribute value at surface boundaries such as
``Reshard.storage``; it is not a ``TensorType.storage`` value.

See [types §2](docs/spec/types.md#2-tensortype).
"""

from __future__ import annotations

from enum import IntEnum


class StorageKind(IntEnum):
    """Memory-space level (backend-generic)."""

    HOST = 1
    GMEM = 2
    SMEM = 3
    RMEM = 4
    TMEM = 5
    UMAT = 6

    def __str__(self) -> str:

        return self.name.lower()


def resolve_storage(value: "str | StorageKind | None") -> "StorageKind | None":
    """Normalise an optional storage spec at a surface boundary.

    ``None`` and ``StorageKind`` pass through; a canonical short name string
    maps to its matching ``StorageKind``. The comparison is deliberately
    case-sensitive so the surface vocabulary follows the enum spellings.
    """
    if value is None or isinstance(value, StorageKind):
        return value
    if isinstance(value, str):
        for kind in StorageKind:
            if str(kind) == value:
                return kind
        names = tuple(str(kind) for kind in StorageKind)
        raise ValueError(
            f"unknown storage {value!r}; expected one of {names} or a StorageKind"
        )
    raise TypeError(f"storage must be a StorageKind, str, or None, got {type(value).__name__}")


__all__ = ["StorageKind", "resolve_storage"]
