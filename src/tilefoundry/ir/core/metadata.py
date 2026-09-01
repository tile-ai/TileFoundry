from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class IRMetadata:
    """Base class for typed metadata attached to an IR expression."""


@dataclass(frozen=True)
class BindingMetadata(IRMetadata):
    """The authored SSA binding name for an expression."""

    name: str


@dataclass(frozen=True)
class SourceSpanMetadata(IRMetadata):
    """Source location for a parser-authored expression."""

    file: str
    line: int
    column: int
    end_line: int | None = None
    end_column: int | None = None


def get_metadata[T: IRMetadata](expr: "Expr", cls: type[T]) -> T | None:
    """Return the metadata whose concrete class is ``cls``, if present."""
    return next((value for value in expr.metadata if type(value) is cls), None)


def binding_name(expr: "Expr") -> str | None:
    """Return the authored SSA binding name attached to ``expr``."""
    binding = get_metadata(expr, BindingMetadata)
    return binding.name if binding is not None else None


def _span_text(expr: "Expr") -> str | None:
    span = get_metadata(expr, SourceSpanMetadata)
    return f"{span.file}:{span.line}:{span.column}" if span is not None else None


def diagnostic_location(expr: "Expr") -> str | None:
    """Return the most precise source identity available for diagnostics."""
    return _span_text(expr) or binding_name(expr)


def value_label(expr: "Expr") -> str | None:
    """The name a report calls one value, stable across runs.

    An authored line disambiguates two values that share a name, and points the
    reader at what produced the row; a definition-order suffix can do neither.
    A declaration carries no span and needs none: its own name is unique.
    """
    name = getattr(expr, "name", None) or binding_name(expr)
    span = get_metadata(expr, SourceSpanMetadata)
    if span is None:
        return name
    return f"{name}:{span.line}" if name else f"<value>:{span.line}"


def value_labels(exprs: "Iterable[Expr]") -> list[str]:
    """One label per expr, unique within the group and located where it can be.

    Uniqueness is a property of the group, not of one value, so it is settled
    here rather than in ``value_label``. A Call carries the printer's SSA name
    and its authored line, which already separate them. Two values can still
    collide: a loop's carried argument keeps its authored name and has no span,
    so two loops accumulating into ``acc`` produce one label twice. Those take a
    numeric suffix in definition order, the only thing left to tell them apart.
    """
    bases = [value_label(expr) or f"<value {index}>" for index, expr in enumerate(exprs)]
    repeated = {base for base, count in Counter(bases).items() if count > 1}
    seen: Counter[str] = Counter()
    labels: list[str] = []
    for base in bases:
        if base not in repeated:
            labels.append(base)
            continue
        seen[base] += 1
        labels.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return labels


def source_metadata(expr: "Expr") -> tuple[IRMetadata, ...]:
    """Copy only authored binding/span metadata from ``expr``."""
    return tuple(
        value for value in expr.metadata if type(value) in {BindingMetadata, SourceSpanMetadata}
    )


def describe_expr(expr: "Expr") -> str:
    """One diagnostic line locating *expr* in authored source."""
    where = _span_text(expr)
    prefix = f"{where}: " if where is not None else ""
    op = type(expr.target).__name__ if hasattr(expr, "target") else type(expr).__name__
    return f"{prefix}binding={binding_name(expr) or '<unnamed>'} op={op}"


def attach_metadata(expr: "Expr", value: IRMetadata) -> None:
    """Attach *value* to *expr* in place, replacing its concrete metadata type."""
    kept = tuple(item for item in expr.metadata if type(item) is not type(value))
    expr.metadata = (*kept, value)


def detach_metadata(expr: "Expr", cls: type[IRMetadata]) -> None:
    """Remove metadata of concrete type *cls* from *expr* in place."""
    expr.metadata = tuple(item for item in expr.metadata if type(item) is not cls)


def remove_metadata(expr: "Expr", cls: type[IRMetadata]) -> "Expr":
    """Return ``expr`` without metadata whose concrete class is ``cls``."""
    updated = tuple(value for value in expr.metadata if type(value) is not cls)
    if updated == expr.metadata:
        return expr
    return replace(expr, metadata=updated)


__all__ = [
    "IRMetadata",
    "BindingMetadata",
    "SourceSpanMetadata",
    "binding_name",
    "describe_expr",
    "diagnostic_location",
    "value_label",
    "value_labels",
    "attach_metadata",
    "detach_metadata",
    "get_metadata",
    "remove_metadata",
    "source_metadata",
]
