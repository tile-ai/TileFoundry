"""Define backend-generic memory-space levels.

Surface strings normalize to ``StorageKind`` at the IR boundary. ``None``
means no memory space; ``UMAT`` marks compiler-internal values whose residency
has not yet been materialized.

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


_STORAGE_NAMES = {
    "host": StorageKind.HOST,
    "gmem": StorageKind.GMEM,
    "smem": StorageKind.SMEM,
    "rmem": StorageKind.RMEM,
    "tmem": StorageKind.TMEM,
}


def resolve_storage(value: "str | StorageKind | None") -> "StorageKind | None":
    """Normalise a storage spec to ``StorageKind | None`` at a surface boundary.

    ``None`` and ``StorageKind`` pass through; a canonical short name string
    (``host`` / ``gmem`` / ``smem`` / ``rmem`` / ``tmem``) maps to its
    ``StorageKind``.
    """
    if value is None or isinstance(value, StorageKind):
        return value
    if isinstance(value, str):
        kind = _STORAGE_NAMES.get(value)
        if kind is None:
            raise ValueError(
                f"unknown storage {value!r}; expected one of "
                f"{sorted(_STORAGE_NAMES)} or a StorageKind"
            )
        return kind
    raise TypeError(f"storage must be a StorageKind, str, or None, got {type(value).__name__}")


__all__ = ["StorageKind", "resolve_storage"]
