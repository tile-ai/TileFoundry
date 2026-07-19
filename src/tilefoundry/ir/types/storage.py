"""Memory-space kinds.

``StorageKind`` is the memory-space *level* a tensor lives in. It is generic
across GPU backends — the backend is the function ``target``, not the storage
name. Device-internal levels (shared / register / tensor memory) are valid only
inside kernel code; whether a particular level is supported is decided by each
target's lowering.

IR never stores storage as a string. A surface (parser / DSL) may accept a
canonical short-name string (``host`` / ``gmem`` / ``smem`` / ``rmem`` /
``tmem``), normalised to ``StorageKind`` (or ``None``) at the boundary via
:func:`resolve_storage` before entering the IR. ``None`` means
"no memory space" — a non-memory-resident compile-time / shape scalar, or an
unspecified op attribute; it is not a user-visible storage value. ``UMAT``
marks an *unmaterialized* value: one with no committed residency yet, to be
materialized to a concrete level before codegen. ``UMAT`` is compiler-internal
— it is created directly (e.g. for source value literals), never accepted as a
surface string, so a runtime annotation cannot carry it.
"""
from __future__ import annotations

from enum import IntEnum


class StorageKind(IntEnum):
    """Memory-space level (backend-generic)."""

    HOST = 1  # host / CPU memory (DLPack kDLCPU)
    GMEM = 2  # device global memory (kDLCUDA / kDLROCM / ...)
    SMEM = 3  # device shared memory (kernel-internal)
    RMEM = 4  # device register memory (kernel-internal)
    TMEM = 5  # device tensor memory (kernel-internal)
    UMAT = 6  # unmaterialized — a value with no committed residency yet
              # (placement-polymorphic); must be materialized before codegen

    def __str__(self) -> str:
        # Surface name (``gmem`` / ``smem`` / ...) rather than the
        # ``IntEnum`` default (the numeric value) so labels / printers
        # render the canonical storage name.
        return self.name.lower()


# Canonical short storage names accepted as string input at a surface boundary
# only (the lowercased ``StorageKind`` member names); never stored in the IR.
# Legacy long aliases (``global`` / ``shared`` / ``reg`` / ``register``) are not
# accepted — use the canonical short name or the ``StorageKind``.
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
    raise TypeError(
        f"storage must be a StorageKind, str, or None, got {type(value).__name__}"
    )


__all__ = ["StorageKind", "resolve_storage"]
