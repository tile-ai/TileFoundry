"""How a record renders as one line of comment.

A record states typed fields ([core-ir §2](docs/spec/core-ir.md#2-expr)); what
they look like is decided here, so a family arrives by declaring what it has
rather than by writing a form string -- which is how five came to spell one value
five ways. Only a value Python cannot already read needs a rendering: three do,
and ``int`` / ``str`` / a mapping of them do not. The constants below are the
ladder of [inspection §2.8](docs/spec/inspection.md#28-record-comment-forms),
one each, and nothing else in the tree spells one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import get_type_hints

from tilefoundry.analysis.metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TimelineSummaryMetadata,
)
from tilefoundry.ir.core.metadata import IRMetadata, SourceSpanMetadata
from tilefoundry.ir.core.values import TotalAndPerUnit, TripInterval
from tilefoundry.visitor_registry.contexts import TrafficBytes

PAIR = "/"
PER_UNIT = "@"
ENTRY = ":"
ENTRIES = ","
FIELD = "="
FIELDS = " "
PARTS = "; "
TRIPS = "*"


def _trip_interval(value: TripInterval, render: Callable[[object], str]) -> str:
    """One interval, offset by the trip index when it repeats."""
    if value.trips <= 1:
        return f"[{value.start}{ENTRIES}{value.end})"
    offset = f"{value.stride}t+"
    return f"[{offset}{value.start}{ENTRIES}{offset}{value.end}){TRIPS}{value.trips}"


RENDER: dict[type, Callable[..., str]] = {
    TrafficBytes: lambda value, render: f"r{value.read}{PAIR}w{value.write}",
    TotalAndPerUnit: (
        lambda value, render: f"{render(value.total)}{PER_UNIT}{render(value.per_unit)}"
    ),
    TripInterval: _trip_interval,
}


def render_value(value: object) -> str:
    """One value, rendered by its own type or as the mapping it is."""
    for value_type, render in RENDER.items():
        if isinstance(value, value_type):
            return render(value, render_value)
    entries = _entries_of(value)
    if entries is not None:
        return ENTRIES.join(f"{key}{ENTRY}{render_value(item)}" for key, item in entries)
    return str(value)


def _entries_of(value: object) -> tuple[tuple[object, object], ...] | None:
    """*value* as key/value pairs, or ``None`` when it is not a mapping."""
    if isinstance(value, Mapping):
        return tuple(value.items())
    if isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 for item in value
    ):
        return value
    return None


_UNSET = object()


@dataclass(frozen=True)
class Projection:
    """One key a comment emits: its name, its type, and where its value is read.

    A field is the identity projection of itself. Anything else -- a sum, a
    count, several fields folded into one value -- is declared here, which is
    what keeps the next one from appearing unannounced.

    *default* is the value this key says nothing by, so it is left out. Without
    one, an empty mapping says nothing and every other value is emitted.
    """

    key: str
    type: object
    of: Callable[..., object]
    default: object = _UNSET
    opt_in: bool = False


@dataclass(frozen=True)
class RecordComment:
    """What one record type emits, in order, under which family name."""

    family: str
    emissions: tuple[Projection, ...]


_COMMENTS: dict[type[IRMetadata], RecordComment] = {}


def _derived_family(record: type) -> str:
    """The family name the class name already states."""
    stem = record.__name__.removesuffix("Metadata")
    return re.sub(r"(?<!^)(?=[A-Z])", "-", stem).lower()


def _field_projections(record: type[IRMetadata], names: tuple[str, ...]) -> tuple[Projection, ...]:
    hints = get_type_hints(record)
    defaults = {field.name: field.default for field in fields(record)}
    return tuple(
        Projection(name, hints[name], _read(name), defaults[name]) for name in names
    )


def _read(name: str) -> Callable[..., object]:
    return lambda record: getattr(record, name)


def comment(
    record: type[IRMetadata],
    *emitted: str | Projection,
    family: str | None = None,
) -> None:
    """Declare that *record* renders as a comment, and what it emits.

    A field is named by its own name; anything else is a ``Projection``. Naming
    nothing emits every field in declaration order. A record with no declaration
    renders as ``None``, which is how metadata that is not a report stays out of
    the comment.

    *family* is only for a record whose reported name is not the one its class
    name states.
    """
    declared = emitted or tuple(field.name for field in fields(record))
    emissions = tuple(
        item
        if isinstance(item, Projection)
        else _field_projections(record, (item,))[0]
        for item in declared
    )
    _COMMENTS[record] = RecordComment(
        family=family or _derived_family(record), emissions=emissions
    )


def comment_of(record: type[IRMetadata]) -> RecordComment | None:
    """What *record* declared it emits, if it declared anything."""
    return _COMMENTS.get(record)


def declared_records() -> tuple[type[IRMetadata], ...]:
    """Every record type that declared it renders as a comment."""
    return tuple(_COMMENTS)


def family_of(record: type[IRMetadata]) -> str:
    """The family name *record* is reported under."""
    declared = _COMMENTS.get(record)
    return declared.family if declared is not None else _derived_family(record)


def render_comment(
    record: IRMetadata, *, opt_in: frozenset[str] = frozenset()
) -> str | None:
    """One record as one comment, or ``None`` when it does not report.

    A key is its declared name with ``_`` written as ``-``; a unit belongs in
    that name, which is where it is said once. A record declaring one key uses
    the family name as that key, because for a record of one thing the family
    and the key say the same thing twice.
    """
    declared = _COMMENTS.get(type(record))
    if declared is None:
        return None
    alone = len(declared.emissions) == 1
    emitted: list[str] = []
    for emission in declared.emissions:
        if emission.opt_in and emission.key not in opt_in:
            continue
        value = emission.of(record)
        if _says_nothing(value, emission.default):
            continue
        key = declared.family if alone else emission.key.replace("_", "-")
        emitted.append(f"{key}{FIELD}{render_value(value)}")
    if alone and emitted:
        return emitted[0]
    return FIELDS.join([declared.family, *emitted])


def _says_nothing(value: object, default: object) -> bool:
    """Whether this value adds nothing to the comment it would appear in."""
    if default is not _UNSET:
        return value == default
    entries = _entries_of(value)
    return entries is not None and not entries


def _paired_flops(record: ComputeCostMetadata) -> dict[str, TotalAndPerUnit[int]]:
    """Each dtype's work, whole and per unit, as one value."""
    per_unit = dict(record.flops_per_unit)
    return {
        dtype: TotalAndPerUnit(total, per_unit.get(dtype, 0))
        for dtype, total in record.flops
    }


def _paired_traffic(
    record: ComputeCostMetadata,
) -> dict[str, TotalAndPerUnit[TrafficBytes]]:
    """Each level's traffic, whole and per unit, as one value."""
    return {
        level: TotalAndPerUnit(moved, record.traffic_per_unit_at(level))
        for level, moved in record.traffic
    }


def _by_operand(record: ComputeCostMetadata) -> dict[str, TrafficBytes]:
    """What each operand moved, positional against ``(*call.args, call)``."""
    last = len(record.operands) - 1
    return {
        ("result" if index == last else str(index)): moved
        for index, moved in enumerate(record.operands)
    }


def _peak_bytes(record: MemoryMetadata) -> dict[str, int]:
    """How much of each level the function holds at its peak."""
    return {item.level: item.peak_bytes for item in record.footprint}


def _persistent_bytes(record: MemoryMetadata) -> int:
    """The part of the peak the function cannot reclaim, across levels."""
    return sum(item.persistent_bytes for item in record.footprint)


def _advisory_count(record: MemoryMetadata) -> int:
    """How many advisories there are; the comment is not where they are read."""
    return len(record.advisories)


def _interval(record: TimelineMetadata) -> TripInterval:
    """The occurrence's interval, with its repetition folded in."""
    return TripInterval(record.start_ns, record.end_ns, record.stride_ns, record.trips)


def _source_span(record: SourceSpanMetadata) -> str:
    """Where the expression was authored, as one location."""
    return f"{record.file}:{record.line}:{record.column}"


comment(
    ComputeCostMetadata,
    Projection("flops", dict[str, TotalAndPerUnit[int]], _paired_flops),
    Projection("traffic", dict[str, TotalAndPerUnit[TrafficBytes]], _paired_traffic),
    Projection("operands", dict[str, TrafficBytes], _by_operand, opt_in=True),
)
comment(
    MemoryMetadata,
    Projection("peak", dict[str, int], _peak_bytes),
    Projection("persistent", int, _persistent_bytes, default=0),
    Projection("advisories", int, _advisory_count, default=0),
)
comment(RooflineMetadata, "ideal_ns", "bound_by")
comment(TimelineMetadata, Projection("interval", TripInterval, _interval))
comment(TimelineSummaryMetadata, family="timeline")
comment(SourceSpanMetadata, Projection("span", str, _source_span), family="source")


__all__ = [
    "ENTRIES",
    "ENTRY",
    "FIELD",
    "FIELDS",
    "PAIR",
    "PARTS",
    "PER_UNIT",
    "RENDER",
    "TRIPS",
    "Projection",
    "RecordComment",
    "comment",
    "comment_of",
    "declared_records",
    "family_of",
    "render_comment",
    "render_value",
]
