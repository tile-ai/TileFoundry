"""Shared matching, resolution, and rendering helpers for IR patterns."""

from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.clause.layout import is_layout_wildcard
from tilefoundry.ir.types import ComposedLayout
from tilefoundry.ir.types.dim import DimFloorDiv, DimMul, DimVar, is_dim_op_call
from tilefoundry.ir.types.substitute import DimSubstitutionError, substitute_shape_dim

UNNAMED_PLACE = "_"
ARRANGEMENT = "every arrangement"
ABSENT = type("Absent", (), {"__repr__": lambda self: "ABSENT"})()
OPAQUE = object()


@dataclass(frozen=True)
class Match:
    """The bindings produced by a successful match."""

    captures: dict = field(default_factory=dict)


def _pattern_type():
    from .pattern import Pattern  # noqa: PLC0415 - pattern protocol cycle

    return Pattern


def _named(value):
    """One enumerated field as an author writes it, or None when unstated."""
    return None if value is None else getattr(value, "name", str(value)).lower()


def _extents(bindings) -> dict:
    return {name: value for name, value in dict(bindings or {}).items() if type(value) is int}


def resolved(value, bindings):
    """Resolve symbols and nested patterns under *bindings*."""
    if isinstance(value, _pattern_type()):
        return value.resolve(bindings)
    if isinstance(value, DimVar) or is_dim_op_call(value):
        return substitute_shape_dim(value, _extents(bindings))
    if isinstance(value, tuple):
        return tuple(resolved(item, bindings) for item in value)
    return value


def evaluated(value, captures):
    """Evaluate one symbolic dimension, or return None while it is unresolved."""
    try:
        held = substitute_shape_dim(value, _extents(captures))
    except DimSubstitutionError:
        return None
    return held if type(held) is int else None


def is_symbolic(value) -> bool:
    return isinstance(value, DimVar) or is_dim_op_call(value)


def written_dim(value) -> str:
    if isinstance(value, DimVar):
        return value.name
    if is_dim_op_call(value):
        left, right = (written_dim(arg) for arg in value.args)
        if isinstance(value.target, DimFloorDiv):
            return f"{left}/{right}"
        if isinstance(value.target, DimMul):
            return f"{left}*{right}"
        return f"({left} {type(value.target).__name__} {right})"
    return str(getattr(value, "value", value))


def written_binding(value) -> str:
    return getattr(value, "name", str(value))


def written_bindings(bindings) -> str:
    return ", ".join(f"{name}={written_binding(value)}" for name, value in bindings)


def written_tuple(items) -> str:
    written = ", ".join(items)
    return f"({written},)" if len(items) == 1 else f"({written})"


def written_grouped(modes) -> str:
    if isinstance(modes, tuple):
        return written_tuple(tuple(written_grouped(mode) for mode in modes))
    return written_place(modes)


def written_place(value, name: str = UNNAMED_PLACE) -> str:
    if isinstance(value, _pattern_type()):
        return value.describe(name)
    if is_symbolic(value):
        return written_dim(value)
    return UNNAMED_PLACE if is_layout_wildcard(value) else str(value)


def written_field(value) -> str | None:
    if isinstance(value, _pattern_type()):
        return value.describe()
    return _named(value)


def written_alternatives(items, name: str = UNNAMED_PLACE) -> str:
    held = tuple(
        (written_bindings(bindings), written_place(alternative, name))
        for bindings, alternative in items
    )
    width = max((len(label) for label, _ in held), default=0)
    lines = []
    for label, written in held:
        first, *rest = written.splitlines() or ("",)
        lines.append(first if not width else f"{label.ljust(width)}  {first}")
        lines.extend(line if not width else f"{' ' * (width + 2)}{line}" for line in rest)
    return "\n".join(lines)


def alternatives_of(pattern, bindings=()) -> tuple:
    if isinstance(pattern, _pattern_type()):
        return pattern.alternatives(bindings)
    return ((tuple(bindings), pattern),)


def matched(pattern, subject, captures=None) -> Match | None:
    """Match a nested pattern, symbolic dimension, wildcard, or fixed value."""
    held = Match(dict(captures or {}))
    if isinstance(pattern, DimVar):
        if pattern.name in held.captures:
            return held if held.captures[pattern.name] == subject else None
        if type(subject) is not int or not pattern.lo <= subject < pattern.hi:
            return None
        return Match({**held.captures, pattern.name: subject})
    if is_dim_op_call(pattern):
        found = evaluated(pattern, held.captures)
        return held if found is not None and found == subject else None
    if isinstance(pattern, _pattern_type()):
        return pattern.match(subject, held.captures)
    return held if fits(pattern, subject) else None


def fits(wanted, held) -> bool:
    """Whether one pattern position admits one concrete value."""
    if wanted is None:
        return True
    if isinstance(wanted, _pattern_type()) or is_symbolic(wanted):
        return matched(wanted, held) is not None
    if is_layout_wildcard(wanted):
        return type(held) is int
    return wanted == held


def grouping(modes):
    """The tuple nesting of *modes*, with the concrete modes erased."""
    if isinstance(modes, tuple):
        return tuple(grouping(mode) for mode in modes)
    return None


def relations_of(values) -> tuple[str, ...]:
    lines: dict[str, None] = {}
    for value in values:
        if isinstance(value, _pattern_type()):
            lines.update(dict.fromkeys(value.relations()))
    return tuple(lines)


def between_rules(op_type) -> tuple:
    return tuple(getattr(op_type, "between", ()))


def refusals_between(op_type, operands: dict) -> tuple[str, ...]:
    return tuple(
        rule.refused(operands) for rule in between_rules(op_type) if not rule.holds(operands)
    )


def affine_frame(layout) -> tuple | None:
    """Read a bare layout or a composition through no transform as offset + layout."""
    if not isinstance(layout, ComposedLayout):
        return 0, layout
    if layout.inner is not None:
        return None
    return layout.offset, layout.outer


__all__ = [
    "ABSENT",
    "ARRANGEMENT",
    "Match",
    "OPAQUE",
    "UNNAMED_PLACE",
    "_named",
    "affine_frame",
    "alternatives_of",
    "between_rules",
    "evaluated",
    "fits",
    "grouping",
    "is_symbolic",
    "matched",
    "refusals_between",
    "relations_of",
    "resolved",
    "written_alternatives",
    "written_binding",
    "written_bindings",
    "written_dim",
    "written_field",
    "written_grouped",
    "written_place",
    "written_tuple",
]
