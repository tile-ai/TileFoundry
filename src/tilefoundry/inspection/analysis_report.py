"""Renderings of what an analysis found.

An analysis leaves a semantic result and typed records on the IR. What a human
or a tool reads is a *rendering* of those, produced here rather than by the
analyses: a family that formatted its own text would be deciding presentation,
and two families would then disagree about it.

Text and JSON are two renderings of one report, built once. That is what makes
them carry the same conclusions rather than two independently maintained views
that drift.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TrafficBytes,
)
from tilefoundry.analysis.api import AnalysisResult
from tilefoundry.analysis.registry import ANALYSES
from tilefoundry.analysis.walk import postorder
from tilefoundry.ir.core import Call, IRMetadata, binding_name, get_metadata
from tilefoundry.ir.hir.function import Function


def _traffic(traffic: tuple[tuple[str, TrafficBytes], ...]) -> dict[str, dict[str, int]]:
    return {
        level: {"read_bytes": value.read_bytes, "write_bytes": value.write_bytes}
        for level, value in traffic
    }


def _compute_cost(record: ComputeCostMetadata) -> dict[str, object]:
    return {
        "flops": dict(record.flops),
        "traffic": _traffic(record.traffic),
        "execution_count": record.execution_count,
    }


def _roofline(record: RooflineMetadata) -> dict[str, object]:
    return {
        "compute_ns": record.compute_ns,
        "memory_ns": record.memory_ns,
        "theoretical_ns": record.theoretical_ns,
        "bound_by": record.bound_by,
    }


def _timeline(record: TimelineMetadata) -> dict[str, object]:
    return {
        "grid_units": record.grid_units,
        "waves": record.waves,
        "start_ns": record.start_ns,
        "end_ns": record.end_ns,
    }


def _memory(record: MemoryMetadata) -> dict[str, object]:
    return {
        "footprint": [
            {
                "level": item.level,
                "peak_bytes": item.peak_bytes,
                "persistent_bytes": item.persistent_bytes,
                "capacity_bytes": item.capacity_bytes,
            }
            for item in record.footprint
        ],
        "traffic": _traffic(record.traffic),
        "lifetimes": [
            {
                "binding": item.binding,
                "level": item.level,
                "bytes": item.bytes,
                "defined_at": item.defined_at,
                "last_used_at": item.last_used_at,
                "persistent": item.persistent,
            }
            for item in record.lifetimes
        ],
        "advisories": list(record.advisories),
    }


# One renderer per record type, so adding a family adds an entry rather than a
# branch in every format.
_RECORDS: tuple[tuple[str, type[IRMetadata], object], ...] = (
    ("compute-cost", ComputeCostMetadata, _compute_cost),
    ("memory", MemoryMetadata, _memory),
    ("roofline", RooflineMetadata, _roofline),
    ("timeline", TimelineMetadata, _timeline),
)


def _records_of(expr: object, selected: frozenset[type[IRMetadata]]) -> dict[str, object]:
    """Every selected record on *expr*, keyed by the family that owns it."""
    result: dict[str, object] = {}
    for name, metadata_type, project in _RECORDS:
        if metadata_type not in selected:
            continue
        record = get_metadata(expr, metadata_type)
        if record is not None:
            result[name] = project(record)
    return result


def _call_label(call: Call, index: int) -> str:
    return binding_name(call) or f"<value {index}>"


def selected_types(
    results: Sequence[AnalysisResult],
) -> tuple[type[IRMetadata], ...]:
    """Which record types these results may be shown through.

    A requested root pulls its dependencies in, so records land on the IR that
    nobody asked to see. What is shown is what the *requested* analyses own,
    intersected with what was actually written: ownership comes from the
    registrations rather than a hand-kept table, and the intersection keeps a
    reader from looking for a record that is not there.

    Every rendering of one run goes through here, so the report and the annotated
    IR cannot make this choice differently.
    """
    if not results:
        return ()
    target = results[0].module.resolve_target()
    owned: set[type[IRMetadata]] = set()
    for item in results:
        owned.update(ANALYSES.resolve(target, item.analysis).produces)
    order: list[type[IRMetadata]] = []
    for item in results:
        for metadata_type in item.metadata_types:
            if metadata_type in owned and metadata_type not in order:
                order.append(metadata_type)
    return tuple(order)


def report(results: Sequence[AnalysisResult]) -> dict[str, object]:
    """One report over every analysis run against one function.

    Only the record types the calls actually wrote are read, so a renderer is
    never sent looking for records that are not there.
    """
    if not results:
        raise ValueError("an analysis report needs at least one result")
    first = results[0]
    function = first.function
    if any(item.function is not function for item in results):
        raise ValueError(
            "an analysis report covers one function; these results cover several"
        )
    selected = frozenset(selected_types(results))
    executed: list[str] = []
    for item in results:
        for selector in item.executed:
            if selector not in executed:
                executed.append(selector)
    target = first.module.resolve_target()
    calls = []
    for index, expr in enumerate(postorder(function.body)):
        if not isinstance(expr, Call):
            continue
        records = _records_of(expr, selected)
        if records:
            calls.append({"value": _call_label(expr, index), **records})
    data = {
        "target": getattr(target, "name", type(target).__name__),
        "module": first.module.name,
        "function": function.name,
        "requested": [item.analysis for item in results],
        "executed": executed,
        "function_records": _records_of(function, selected),
        "calls": calls,
    }
    if ComputeCostMetadata in selected:
        data["totals"] = _work_totals(function)
    return data


def _work_totals(function: Function) -> dict[str, object]:
    """The function's work, summed from the per-Call records.

    Compute cost has no whole-function record precisely because this sum is all
    there would be in one: adding flops and bytes is exact, so recording the
    result would be storing a second copy of it. Summing is therefore the one
    thing this module computes, and it computes nothing else.
    """
    flops: dict[str, int] = {}
    traffic: dict[str, TrafficBytes] = {}
    for expr in postorder(function.body):
        record = get_metadata(expr, ComputeCostMetadata)
        if record is None:
            continue
        for name, value in record.flops:
            flops[name] = flops.get(name, 0) + value
        for level, value in record.traffic:
            current = traffic.get(level, TrafficBytes())
            traffic[level] = TrafficBytes(
                current.read_bytes + value.read_bytes,
                current.write_bytes + value.write_bytes,
            )
    return {
        "flops": dict(sorted(flops.items())),
        "traffic": _traffic(tuple(sorted(traffic.items()))),
    }


def _flop_text(flops: dict[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in sorted(flops.items())) or "0"


def _traffic_text(traffic: dict[str, dict[str, int]]) -> str:
    return (
        ", ".join(
            f"{level}=r{value['read_bytes']}/w{value['write_bytes']}"
            for level, value in sorted(traffic.items())
        )
        or "0"
    )


def render_text(data: dict[str, object]) -> str:
    """The report as one stable line per conclusion, each prefixed with ``#``.

    A conclusion appears only when a record states it, or -- for the work totals
    -- when it is the exact sum of records that do.
    """
    lines = [
        f"analysis target={data['target']} module={data['module']} "
        f"function={data['function']}",
        f"analyses={','.join(data['requested'])} executed={','.join(data['executed'])}",
    ]
    totals = data.get("totals")
    if totals is not None:
        lines.append(f"flops {_flop_text(totals['flops'])}")
        lines.append(f"traffic {_traffic_text(totals['traffic'])}")
    records = data["function_records"]
    if "memory" in records:
        memory = records["memory"]
        lines.append(
            "peak-footprint "
            + (
                ", ".join(
                    f"{item['level']}={item['peak_bytes']}"
                    for item in memory["footprint"]
                )
                or "0"
            )
        )
        lines.extend(f"advisory {note}" for note in memory["advisories"])
    if "roofline" in records:
        bound = records["roofline"]
        lines.append(
            f"theoretical-bound={bound['theoretical_ns']}ns by={bound['bound_by']}"
        )
    if "timeline" in records:
        lines.append(f"theoretical-makespan={records['timeline']['end_ns']}ns")
    for call in data["calls"]:
        parts = [f"value={call['value']}"]
        if "compute-cost" in call:
            cost = call["compute-cost"]
            parts.append(
                "flops="
                + (
                    ",".join(
                        f"{name}:{value}"
                        for name, value in sorted(cost["flops"].items())
                    )
                    or "0"
                )
            )
        if "roofline" in call:
            parts.append(f"bound={call['roofline']['theoretical_ns']}ns")
        if "timeline" in call:
            timeline = call["timeline"]
            parts.append(
                f"placement={timeline['start_ns']}-{timeline['end_ns']}ns "
                f"units={timeline['grid_units']} waves={timeline['waves']}"
            )
        lines.append(" ".join(parts))
    return "\n".join(f"# {line}" for line in lines)


def render_json(data: dict[str, object]) -> str:
    """The same report as JSON, sorted so two runs compare byte for byte."""
    return json.dumps(data, indent=2, sort_keys=True)


__all__ = ["render_json", "render_text", "report", "selected_types"]
