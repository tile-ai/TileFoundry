"""Index dialect operations and exported types by namespace path and name."""

from __future__ import annotations

from collections.abc import Iterable

from .op_schema import OpSchema

__all__ = [
    "VALID_DIALECTS",
    "OpRegistry",
    "children",
    "get_op_by_name",
    "get_stmt_by_name",
    "iter_schemas",
    "lookup",
    "own_schema_of",
    "registry",
    "schema_of",
]

VALID_DIALECTS = ("tf", "T")


def admit(table: dict, dialect: str, path: tuple[str, ...], name: str, entry: OpSchema | type) -> None:
    """Add one entry to *table*, refusing a name that is already owned or invalid."""
    if dialect not in VALID_DIALECTS:
        raise ValueError(f"dialect must be one of {VALID_DIALECTS!r}, got {dialect!r}")
    if not isinstance(name, str) or not name or "." in name or any(not isinstance(part, str) or not part or "." in part for part in path):
        raise ValueError("registry names and namespace components must be non-empty")
    for depth, part in enumerate((*path, name)):
        existing = table.get((dialect, path[:depth], part))
        if existing is not None:
            owner = existing.builder if isinstance(existing, OpSchema) else existing
            raise ValueError(
                f"{(dialect, path[:depth], part)!r} is already owned by "
                f"{owner.__module__}.{owner.__qualname__}"
            )
    prefix = (*path, name)
    for (d, namespace, _), existing in table.items():
        if d == dialect and namespace[:len(prefix)] == prefix:
            owner = existing.builder if isinstance(existing, OpSchema) else existing
            raise ValueError(
                f"{(dialect, prefix)!r} is already owned by namespace of "
                f"{owner.__module__}.{owner.__qualname__}"
            )
    table[dialect, path, name] = entry


class OpRegistry:
    """The records of every registered Op class and declaration type.

    ``entries`` maps ``(dialect, path, name)`` to an ``OpSchema`` or an exported
    type, in registration order. ``class_keys`` and ``class_aliases`` map a class
    to its canonical key and its alias keys. ``schemas`` maps ``id`` of a class
    or an alias builder to the schema registered for it.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.entries: dict[tuple[str, tuple[str, ...], str], OpSchema | type] = {}
        self.class_aliases: dict[type, tuple[tuple[str, tuple[str, ...], str], ...]] = {}
        self.class_keys: dict[type, tuple[str, tuple[str, ...], str]] = {}
        self.schemas: dict[int, OpSchema] = {}

    def owns(self, cls: type) -> bool:
        """Whether *cls* is registered as an Op class or an exported type."""
        return any(
            entry is cls or isinstance(entry, OpSchema) and entry.op_class is cls
            for entry in self.entries.values()
        )

    def add_operation(
        self, dialect: str, path: tuple[str, ...], cls: type, schemas: list[OpSchema]
    ) -> None:
        """Register *cls* with its canonical schema first and one schema per alias after it.

        Either every schema is admitted or none is.
        """
        pending = dict(self.entries)
        for schema in schemas:
            admit(pending, dialect, path, schema.name, schema)
        self.entries.update(pending)
        for schema in schemas[1:]:
            self.schemas[id(schema.builder)] = schema
        self.class_keys[cls] = (dialect, path, schemas[0].name)
        self.class_aliases[cls] = tuple((dialect, path, schema.name) for schema in schemas[1:])
        self.schemas[id(cls)] = schemas[0]

    def add_type(self, dialect: str, path: tuple[str, ...], cls: type) -> None:
        """Register the declaration type *cls* under its own name."""
        pending = dict(self.entries)
        admit(pending, dialect, path, cls.__name__, cls)
        self.entries.update(pending)
        self.class_keys[cls] = (dialect, path, cls.__name__)


registry = OpRegistry()


def schema_of(obj: object) -> OpSchema | None:
    """The schema registered for *obj*, a class or an alias builder.

    A class inherits the schema of its nearest registered base.
    """
    for held in obj.__mro__ if isinstance(obj, type) else (obj,):
        schema = registry.schemas.get(id(held))
        if schema is not None:
            return schema
    return None


def own_schema_of(cls: type) -> OpSchema | None:
    """The schema registered for *cls* itself, without looking at its bases."""
    return registry.schemas.get(id(cls))


def lookup(dialect: str, path: tuple[str, ...], name: str) -> OpSchema | type | None:
    """Return the single record at a dialect path."""
    return registry.entries.get((dialect, path, name))


def children(dialect: str, path: tuple[str, ...]) -> list[str]:
    """List records and child namespaces immediately under a dialect path."""
    return sorted({
        namespace[len(path)] if len(namespace) > len(path) else name
        for (d, namespace, name) in registry.entries
        if d == dialect and namespace[:len(path)] == path
    })


def iter_schemas() -> Iterable[OpSchema]:
    """Iterate callable schemas in registration order, excluding exported types."""
    return (entry for entry in registry.entries.values() if isinstance(entry, OpSchema))


def first_schema(dialect: str, name: str) -> OpSchema | None:
    """The callable schema at the top level of *dialect* named *name*, if any."""
    entry = lookup(dialect, (), name)
    return entry if isinstance(entry, OpSchema) else None


def get_op_by_name(name: str) -> type | None:
    """Return the HIR operation class for a top-level name, excluding aliases."""
    schema = first_schema("tf", name)
    return schema.op_class if schema is not None else None


def get_stmt_by_name(name: str) -> type | None:
    """Return the TIR operation class for a top-level name, excluding aliases."""
    schema = first_schema("T", name)
    return schema.op_class if schema is not None else None
