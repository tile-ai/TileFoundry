"""Typed metadata records an IR expression carries, and their diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import ClassVar


@dataclass(frozen=True)
class IRMetadata:
    """Base class for typed metadata attached to an IR expression.

    ``family`` names the analysis report family a record belongs to; ``None``
    keeps the record out of the report.
    """

    family: ClassVar[str | None] = None

    def resolve(self, ctx) -> IRMetadata:
        """Interpret this record in the current visit context."""
        return self


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


@dataclass(frozen=True)
class RangeMetadata(IRMetadata):
    """Conservative half-open value bounds ``[lo, hi)`` for one expression."""

    lo: int
    hi: int


class Annotated:
    """A node whose ``metadata`` tuple holds at most one record of each concrete class."""

    def attach_metadata(self, value: IRMetadata) -> None:
        """Attach *value* in place, replacing the record of its concrete type."""
        kept = tuple(item for item in self.metadata if type(item) is not type(value))
        self.metadata = (*kept, value)

    def detach_metadata(self, cls: type[IRMetadata]) -> None:
        """Remove the metadata of concrete type *cls* in place."""
        self.metadata = tuple(item for item in self.metadata if type(item) is not cls)


def get_metadata[T: IRMetadata](expr: "Expr", cls: type[T], *, ctx=None) -> T | None:
    """Read a definition record, optionally interpreted in a visit context."""
    record = next((value for value in expr.metadata if type(value) is cls), None)
    return record if record is None or ctx is None else record.resolve(ctx)


def binding_name(expr: "Expr") -> str | None:
    """Return the authored SSA binding name attached to ``expr``."""
    binding = get_metadata(expr, BindingMetadata)
    return binding.name if binding is not None else None


def span_text(expr: "Expr") -> str | None:
    span = get_metadata(expr, SourceSpanMetadata)
    return f"{span.file}:{span.line}:{span.column}" if span is not None else None


def diagnostic_location(expr: "Expr") -> str | None:
    """Return the most precise source identity available for diagnostics."""
    return span_text(expr) or binding_name(expr)


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
    return unique_labels([value_label(expr) or "" for expr in exprs])


def unique_labels(bases: list[str]) -> list[str]:
    """Fill unnamed values and disambiguate repeated names in group order."""
    bases = [base or f"<value {index}>" for index, base in enumerate(bases)]
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


def describe_value(expr: "Expr") -> str:
    """One diagnostic line locating *expr* in authored source."""
    where = span_text(expr)
    prefix = f"{where}: " if where is not None else ""
    op = type(expr.target).__name__ if hasattr(expr, "target") else type(expr).__name__
    return f"{prefix}binding={binding_name(expr) or '<unnamed>'} op={op}"


def remove_metadata(expr: "Expr", cls: type[IRMetadata]) -> "Expr":
    """Return ``expr`` without metadata whose concrete class is ``cls``."""
    updated = tuple(value for value in expr.metadata if type(value) is not cls)
    if updated == expr.metadata:
        return expr
    return replace(expr, metadata=updated)


__all__ = [
    "Annotated",
    "IRMetadata",
    "BindingMetadata",
    "RangeMetadata",
    "SourceSpanMetadata",
    "binding_name",
    "describe_value",
    "diagnostic_location",
    "value_label",
    "value_labels",
    "unique_labels",
    "get_metadata",
    "remove_metadata",
    "source_metadata",
]
