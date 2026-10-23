"""Index dialect operations and exported types by namespace path and name."""

from __future__ import annotations

from typing import Iterable

from tilefoundry.ir.core.op_schema import OpSchema

_VALID_DIALECTS = ("tf", "T")
_REGISTRY: dict[tuple[str, tuple[str, ...], str], OpSchema | type] = {}


def lookup(dialect: str, path: tuple[str, ...], name: str) -> OpSchema | type | None:
    """Return the single record at a dialect path."""
    return _REGISTRY.get((dialect, path, name))


def children(dialect: str, path: tuple[str, ...]) -> list[str]:
    """List records and child namespaces immediately under a dialect path."""
    return sorted({
        namespace[len(path)] if len(namespace) > len(path) else name
        for (d, namespace, name) in _REGISTRY
        if d == dialect and namespace[:len(path)] == path
    })


def _register(dialect: str, path: tuple[str, ...], name: str, entry: OpSchema | type) -> None:
    if dialect not in _VALID_DIALECTS:
        raise ValueError(f"dialect must be one of {_VALID_DIALECTS!r}, got {dialect!r}")
    if not isinstance(name, str) or not name or any(not part for part in path):
        raise ValueError("registry names and namespace components must be non-empty")
    for depth, part in enumerate((*path, name)):
        existing = lookup(dialect, path[:depth], part)
        if existing is not None:
            owner = existing.builder if isinstance(existing, OpSchema) else existing
            raise ValueError(
                f"{'.'.join((dialect, *path[:depth], part))} is already owned by "
                f"{owner.__module__}.{owner.__qualname__}"
            )
    prefix = (*path, name)
    for (d, namespace, _), existing in _REGISTRY.items():
        if d == dialect and namespace[:len(prefix)] == prefix:
            owner = existing.builder if isinstance(existing, OpSchema) else existing
            raise ValueError(
                f"{'.'.join((dialect, *prefix))} is already owned by namespace of "
                f"{owner.__module__}.{owner.__qualname__}"
            )
    _REGISTRY[dialect, path, name] = entry


def iter_schemas() -> Iterable[OpSchema]:
    """Iterate callable schemas in registration order, excluding exported types."""
    return (entry for entry in _REGISTRY.values() if isinstance(entry, OpSchema))


def _first_schema(dialect: str, name: str) -> OpSchema | None:
    entry = lookup(dialect, (), name)
    return entry if isinstance(entry, OpSchema) else None


def get_op_by_name(name: str) -> type | None:
    """Return the HIR operation class for a top-level name, excluding aliases."""
    schema = _first_schema("tf", name)
    return schema.op_class if schema is not None else None


def get_stmt_by_name(name: str) -> type | None:
    """Return the TIR operation class for a top-level name, excluding aliases."""
    schema = _first_schema("T", name)
    return schema.op_class if schema is not None else None


__all__ = ["lookup", "children", "iter_schemas", "get_op_by_name", "get_stmt_by_name"]
