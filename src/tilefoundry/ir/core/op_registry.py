"""OpSchema registry: the ``(dialect, name) -> [OpSchema]`` index every callable Op registers into."""
from __future__ import annotations

from typing import Iterable

_VALID_DIALECTS: tuple[str, ...] = ("tf", "T")

# ``(dialect, name) -> list[OpSchema]`` — sole canonical index.
_schemas_by_dialect_name: "dict[tuple[str, str], list[OpSchema]]" = {}

# --- Schema registration -------------------------------------------------

def _register_schema(schema: "OpSchema", *, prepend: bool = False) -> None:
    """Add an OpSchema to the list-per-name overload registry.

    Called by ``@register_op`` (append) and ``@register_alias``
    (``prepend=True``). Multiple schemas may share the same
    ``(dialect, name)`` for overloading; resolution picks the first
    whose ParamDef patterns match the args (F3 first-match lock).
    Aliases prepend so that schema-aware first-match resolution picks
    the alias over a legacy real-Op schema of the same name, while
    ``op_class``-keyed lookups transparently skip the alias entry.
    Idempotent on the same schema instance.
    """
    key = (schema.dialect, schema.name)
    bucket = _schemas_by_dialect_name.setdefault(key, [])
    if schema in bucket:
        return
    if prepend:
        bucket.insert(0, schema)
    else:
        bucket.append(schema)

# --- Schema queries ------------------------------------------------------

def get_schemas(dialect: str, name: str) -> "list[OpSchema]":
    """Return the list of OpSchema candidates for ``(dialect, name)``."""
    return list(_schemas_by_dialect_name.get((dialect, name), ()))

def iter_schemas() -> "Iterable[OpSchema]":
    """Iterate every registered schema in stable order."""
    for bucket in _schemas_by_dialect_name.values():
        yield from bucket

def iter_schema_names(dialect: str) -> "Iterable[str]":
    """Yield distinct ``name`` values registered under ``dialect``."""
    seen: set[str] = set()
    for (d, n) in _schemas_by_dialect_name:
        if d == dialect and n not in seen:
            seen.add(n)
            yield n

# --- Class-keyed view helpers (derived from schemas) --------------------
#
# These compute their result from ``_schemas_by_dialect_name``; they
# return the ``op_class`` of the **first** registered schema for the
# requested key. With multi-schema overloads the parser does pattern-
# based first-match resolution; these helpers cover the simpler
# "name → class" lookup that earlier code relied on.

def _first_op_class(dialect: str, name: str) -> type | None:
    """Return the first **real-Op** class for ``(dialect, name)``.

    Surface-alias schemas (``schema.op_class is None``) are skipped so
    legacy callers that need a concrete IR class continue to see one
    even when an alias has been prepended for the same name.
    """
    bucket = _schemas_by_dialect_name.get((dialect, name))
    if not bucket:
        return None
    for s in bucket:
        if s.op_class is not None:
            return s.op_class
    return None

def _first_schema(dialect: str, name: str) -> "OpSchema | None":
    """Return the first registered ``OpSchema`` for ``(dialect, name)``.

    Unlike :func:`_first_op_class`, this honours alias prepend order —
    an alias schema (if any) wins over the legacy real-Op schema. Used
    by parser dispatch where the goal is the **schema**, not the IR
    class, so that ``schema.builder`` can construct the right target.
    """
    bucket = _schemas_by_dialect_name.get((dialect, name))
    if not bucket:
        return None
    return bucket[0]

def get_op_by_name(name: str) -> type | None:
    """Return the HIR (``tf``-dialect) Op class for ``name``, or None."""
    return _first_op_class("tf", name)

def get_stmt_by_name(name: str) -> type | None:
    """Return the TIR (``T``-dialect) Op class for ``name``, or None.

    Despite the legacy "stmt" suffix, ``T``-dialect callable units are
    Ops (effect-ful ones placed in Stmt position via
    ``Evaluate(op, args)``).
    """
    return _first_op_class("T", name)

__all__ = [
    "get_op_by_name",
    "get_stmt_by_name",
    "_register_schema",
    "_first_schema",
    "get_schemas",
    "iter_schemas",
    "iter_schema_names",
]
