from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class IRMetadata:
    """Base class for typed metadata attached to an IR expression."""


@dataclass(frozen=True)
class BindingMetadata(IRMetadata):
    """The authored SSA binding name for an expression."""

    name: str


@dataclass(frozen=True)
class ExecutionDomainMetadata(IRMetadata):
    """The Mesh scopes an occurrence was authored inside, outermost first.

    Which participants run an occurrence is a fact about where it was written,
    not about the layout its result happens to carry: a value may be laid out
    across threads while the work that produced it runs on one CTA. The stack is
    kept as authored so a reader can ask about any level of it.
    """

    scopes: tuple["Mesh", ...] = ()

    def at(self, level: str) -> "Mesh | None":
        """The innermost scope naming *level*, or None when none does."""
        for mesh in reversed(self.scopes):
            if any(topology.name == level for topology in mesh.topologies):
                return mesh
        return None


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


def diagnostic_location(expr: "Expr") -> str | None:
    """Return the most precise source identity available for diagnostics."""
    span = get_metadata(expr, SourceSpanMetadata)
    if span is not None:
        return f"{span.file}:{span.line}:{span.column}"
    return binding_name(expr)


def source_metadata(expr: "Expr") -> tuple[IRMetadata, ...]:
    """Copy only authored binding/span metadata from ``expr``."""
    return tuple(
        value for value in expr.metadata if type(value) in {BindingMetadata, SourceSpanMetadata}
    )


def replace_metadata(expr: "Expr", value: IRMetadata) -> "Expr":
    """Return ``expr`` with metadata of ``value``'s concrete class replaced."""
    value_cls = type(value)
    found = False
    updated = []
    for current in expr.metadata:
        if type(current) is value_cls:
            updated.append(value)
            found = True
        else:
            updated.append(current)
    if not found:
        updated.append(value)
    return replace(expr, metadata=tuple(updated))


def remove_metadata(expr: "Expr", cls: type[IRMetadata]) -> "Expr":
    """Return ``expr`` without metadata whose concrete class is ``cls``."""
    updated = tuple(value for value in expr.metadata if type(value) is not cls)
    if updated == expr.metadata:
        return expr
    return replace(expr, metadata=updated)


__all__ = [
    "IRMetadata",
    "BindingMetadata",
    "ExecutionDomainMetadata",
    "SourceSpanMetadata",
    "binding_name",
    "diagnostic_location",
    "get_metadata",
    "remove_metadata",
    "replace_metadata",
    "source_metadata",
]
