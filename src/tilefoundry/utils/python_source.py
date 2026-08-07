"""Small, layer-independent vocabulary for values that render Python source."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, is_dataclass


@dataclass(frozen=True)
class PythonExpr:
    """The imports and expression text that rebuild one value."""

    imports: tuple[str, ...]
    text: str


def _merge_imports(*imports: tuple[str, ...]) -> tuple[str, ...]:
    grouped: dict[str, set[str]] = {}
    other: set[str] = set()
    for statement in (statement for group in imports for statement in group):
        prefix, separator, names = statement.partition(" import ")
        if separator and prefix.startswith("from "):
            grouped.setdefault(prefix[5:], set()).update(
                name.strip() for name in names.split(",")
            )
        else:
            other.add(statement)
    merged = [
        f"from {module} import {', '.join(sorted(names))}"
        for module, names in grouped.items()
    ]
    return tuple(sorted([*merged, *other]))


def _value_to_python(value: object) -> PythonExpr:
    to_python = getattr(value, "to_python", None)
    if callable(to_python):
        return to_python()
    if isinstance(value, str):
        return PythonExpr((), json.dumps(value))
    if value is None or isinstance(value, (bool, int, float)):
        return PythonExpr((), repr(value))
    if isinstance(value, tuple):
        values = tuple(_value_to_python(item) for item in value)
        suffix = "," if len(values) == 1 else ""
        return PythonExpr(
            _merge_imports(*(item.imports for item in values)),
            "(" + ", ".join(item.text for item in values) + suffix + ")",
        )
    raise TypeError(f"cannot render {type(value).__name__} as canonical Python")


def dataclass_to_python(value: object, import_module: str) -> PythonExpr:
    """Render one dataclass through its source-reconstructing value fields."""
    if not is_dataclass(value):
        raise TypeError(f"cannot render non-dataclass {type(value).__name__}")
    value_type = type(value)
    if value_type.__qualname__ != value_type.__name__:
        raise TypeError(
            f"cannot render non-top-level {value_type.__module__}."
            f"{value_type.__qualname__}"
        )
    values = tuple(
        (field.name, _value_to_python(getattr(value, field.name)))
        for field in fields(value)
        if field.init or field.compare
    )
    return PythonExpr(
        _merge_imports(
            (f"from {import_module} import {value_type.__qualname__}",),
            *(rendered.imports for _, rendered in values),
        ),
        f"{value_type.__qualname__}("
        + ", ".join(f"{name}={rendered.text}" for name, rendered in values)
        + ")",
    )


__all__ = ["PythonExpr", "dataclass_to_python"]
