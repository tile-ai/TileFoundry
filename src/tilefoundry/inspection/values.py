"""Canonical metadata comment printer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass

from tilefoundry.ir.core.metadata import IRMetadata
from tilefoundry.ir.core.values import TripInterval

PAIR, PER_UNIT, ENTRY, ENTRIES, FIELD, FIELDS, PARTS, TRIPS = (
    "/",
    "@",
    ":",
    ",",
    "=",
    " ",
    "; ",
    "*",
)

_UNSET = object()


class Prose(str):
    """Sentence text rendered as a quoted DSL string."""


@dataclass(frozen=True)
class ReportIdentity(IRMetadata):
    target: str = ""
    module: str = ""
    function: str = ""
    topology: str = "none"


@dataclass(frozen=True)
class ReportSelection(IRMetadata):
    requested: tuple[str, ...] = ()
    executed: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemorySummary(IRMetadata):
    peak_bytes: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class AdvisorySummary(IRMetadata):
    text: Prose


@dataclass(frozen=True)
class PerformanceSummaryView(IRMetadata):
    root: str = ""
    predicted_ns: int = 0
    waves: int = 1


class CommentPrinter:
    """Render metadata by explicit ``print_<Type>`` methods."""

    def print(self, value):
        method = getattr(self, f"print_{type(value).__name__}", None)
        if method:
            return method(value)
        if is_dataclass(value):
            return ENTRIES.join(
                f"{item.name}{ENTRY}{self.print(getattr(value, item.name))}"
                for item in fields(value)
            )
        if isinstance(value, Mapping):
            return ENTRIES.join(f"{k}{ENTRY}{self.print(v)}" for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return ENTRIES.join(self.print(v) for v in value)
        return str(value)

    def print_Prose(self, value):
        return json.dumps(str(value))

    def print_TrafficBytes(self, value):
        return f"r{value.read}{PAIR}w{value.write}"

    def print_TripInterval(self, value):
        if value.trips <= 1:
            return f"[{value.start},{value.end})"
        offset = f"{value.stride}t+"
        return f"[{offset}{value.start},{offset}{value.end}){TRIPS}{value.trips}"

    @staticmethod
    def _empty_without_default(value):
        if isinstance(value, Mapping | tuple | list):
            return not value
        return False

    def _record(self, family, values):
        out = []
        for item in values:
            key, value = item[:2]
            default = item[2] if len(item) == 3 else _UNSET
            if default is not _UNSET:
                if value == default:
                    continue
            elif self._empty_without_default(value):
                continue
            out.append(f"{key.replace('_', '-')}={self.print(value)}")
        return FIELDS.join([family, *out]) if out else family

    def _single(self, family, value, default=_UNSET):
        if default is not _UNSET and value == default:
            return family
        if default is _UNSET and self._empty_without_default(value):
            return family
        return f"{family}{FIELD}{self.print(value)}"

    def print_ComputeCostMetadata(self, record, **_):
        def spread_values(spread):
            logical = dict(spread.logical)
            total = dict(spread.total)
            per_unit = [dict(value) for value in spread.per_unit]
            names = sorted({*logical, *total, *(name for value in per_unit for name in value)})
            return {
                name: (
                    f"{logical.get(name, 0)}{PAIR}{total.get(name, 0)}"
                    + "".join(
                        f"{PER_UNIT}{level}{ENTRY}{value.get(name, 0)}"
                        for level, value in zip(record.topologies, per_unit, strict=False)
                    )
                )
                for name in names
            }

        return self._record(
            "compute-cost",
            (
                ("flops", spread_values(record.flops)),
                ("other_ops", spread_values(record.other_ops)),
            ),
        )

    def print_TrafficMetadata(self, record, *, opt_in=frozenset()):
        traffic = {
            level: (
                f"{self.print(spread.total)}"
                + (
                    f"{PER_UNIT}"
                    + ENTRIES.join(
                        f"{topology}{ENTRY}{self.print(value)}"
                        for topology, value in zip(record.topologies, spread.per_unit, strict=False)
                    )
                    if spread.per_unit
                    else ""
                )
            )
            for level, spread in record.storage.kinds
        }
        values = [("traffic", traffic)]
        if "operands" in opt_in:
            last = len(record.operands) - 1
            operands = {
                "result" if index == last else str(index): moved
                for index, moved in enumerate(record.operands)
            }
            values.append(("operands", operands))
        return self._record("traffic", values)

    def print_MemoryMetadata(self, record, **_):
        return self._record(
            "memory",
            (
                ("peak", {item.memory_level: item.peak_bytes for item in record.footprint}),
                ("persistent", sum(item.persistent_bytes for item in record.footprint), 0),
                ("advisories", len(record.advisories), 0),
            ),
        )

    def print_LoopFootprintMetadata(self, record, **_):
        footprints = {
            f"{item.buffer}@{item.memory_level}": PAIR.join(
                str(value) for value in (item.bytes, item.device_bytes, item.repeated_bytes)
            )
            for item in record.footprints
        }
        return self._record(
            "loop-footprint",
            (("footprints", footprints), ("status", "complete" if record.known else "lower-bound")),
        )

    def print_RooflineMetadata(self, record, **_):
        return self._record(
            "roofline", (("ideal_ns", record.ideal_ns, 0), ("bound_by", record.bound_by, "none"))
        )

    def print_PerformanceMetadata(self, record, **_):
        interval = TripInterval(
            record.timeline.start_ns,
            record.timeline.end_ns,
            record.timeline.stride_ns,
            record.timeline.trips,
        )
        return self._single("performance", interval)

    def print_PerformanceSummaryMetadata(self, record, **_):
        predicted = record.timeline.end_ns - record.timeline.start_ns
        return self._record("performance", (("predicted_ns", predicted), ("waves", record.waves)))

    def print_SourceSpanMetadata(self, record, **_):
        return self._single("source", f"{record.file}:{record.line}:{record.column}")

    def print_ReportIdentity(self, record, **_):
        return self._record(
            "analysis",
            (
                ("target", record.target, ""),
                ("module", record.module, ""),
                ("function", record.function, ""),
                ("topology", record.topology, "none"),
            ),
        )

    def print_ReportSelection(self, record, **_):
        return self._record(
            "selection", (("requested", record.requested, ()), ("executed", record.executed, ()))
        )

    def print_MemorySummary(self, record, **_):
        return self._single("peak-footprint", record.peak_bytes, {})

    def print_AdvisorySummary(self, record, **_):
        return self._single("advisory", record.text)

    def print_PerformanceSummaryView(self, record, **_):
        return self._record(
            "performance",
            (
                ("root", record.root, ""),
                ("predicted_ns", record.predicted_ns, 0),
                ("waves", record.waves),
            ),
        )


_PRINTER = CommentPrinter()


def render_comment(record, *, opt_in=frozenset()):
    method = getattr(_PRINTER, f"print_{type(record).__name__}", None)
    return method(record, opt_in=opt_in) if method else None


def peak_footprint(record):
    return {item.memory_level: item.peak_bytes for item in record.footprint}


__all__ = [
    "CommentPrinter",
    "Prose",
    "render_comment",
    "PAIR",
    "PER_UNIT",
    "ENTRY",
    "ENTRIES",
    "FIELD",
    "FIELDS",
    "PARTS",
    "TRIPS",
    "ReportIdentity",
    "ReportSelection",
    "MemorySummary",
    "AdvisorySummary",
    "PerformanceSummaryView",
    "peak_footprint",
]
