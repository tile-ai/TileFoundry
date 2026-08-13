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
from dataclasses import dataclass

from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineMetadata,
    TimelineSummaryMetadata,
    TrafficBytes,
)
from tilefoundry.analysis.api import AnalysisResult
from tilefoundry.analysis.walk import postorder, tensor_types
from tilefoundry.inspection.python_printer import (
    PythonPrintOptions,
    _PrintedStatement,
    _render_hir_function,
)
from tilefoundry.ir.core import Call, IRMetadata, binding_name, get_metadata
from tilefoundry.ir.hir.function import Function


def _traffic(traffic: tuple[tuple[str, TrafficBytes], ...]) -> dict[str, dict[str, int]]:
    return {
        level: {"read": value.read, "write": value.write}
        for level, value in traffic
    }


def _type_text(type_: object) -> str:
    """One operand's type, short enough to sit on a report line."""
    tensors = tensor_types(type_)
    if not tensors:
        return str(type_)
    first = tensors[0]
    shape = ",".join(str(dim) for dim in first.shape)
    more = "+" if len(tensors) > 1 else ""
    storage = "" if first.storage is None else f" {first.storage}"
    return f"{first.dtype.name}[{shape}]{more}{storage}"


def _operand_name(operand: object) -> str:
    """What to call this operand, from the program rather than from the record."""
    name = binding_name(operand) or getattr(operand, "name", None)
    if name:
        return name
    if isinstance(operand, Call):
        return type(operand.target).__name__.lower()
    return type(operand).__name__.lower()


def _operands(record: ComputeCostMetadata, expr: object) -> list[dict[str, object]]:
    """Each recorded amount, against the operand it was charged to."""
    if not isinstance(expr, Call):
        return []
    operands = (*expr.args, expr)
    return [
        {
            "arg": "result" if index == len(expr.args) else index,
            "name": _operand_name(operand),
            "type": _type_text(operand.type),
            "read": moved.read,
            "write": moved.write,
        }
        for index, (operand, moved) in enumerate(zip(operands, record.operands))
    ]


def _compute_cost(record: ComputeCostMetadata, expr: object) -> dict[str, object]:
    projected: dict[str, object] = {
        "flops": dict(record.flops),
        "flops_per_unit": dict(record.flops_per_unit),
        "traffic": _traffic(record.traffic),
        "traffic_per_unit": _traffic(record.traffic_per_unit),
    }
    operands = _operands(record, expr)
    if operands:
        projected["operands"] = operands
    return projected


def _roofline(record: RooflineMetadata, expr: object) -> dict[str, object]:
    return {
        "compute_ns": record.compute_ns,
        "memory_ns": record.memory_ns,
        "ideal_ns": record.ideal_ns,
        "bound_by": record.bound_by,
    }


def _timeline(record: TimelineMetadata, expr: object) -> dict[str, object]:
    return {
        "start_ns": record.start_ns,
        "end_ns": record.end_ns,
        "trips": record.trips,
        "stride_ns": record.stride_ns,
    }


def _timeline_summary(
    record: TimelineSummaryMetadata, expr: object
) -> dict[str, object]:
    return {
        "local_makespan_ns": record.local_makespan_ns,
        "waves": record.waves,
        "estimated_kernel_ns": record.estimated_kernel_ns,
    }


def _memory(record: MemoryMetadata, expr: object) -> dict[str, object]:
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




_RECORDS: tuple[tuple[str, type[IRMetadata], object], ...] = (
    ("compute-cost", ComputeCostMetadata, _compute_cost),
    ("memory", MemoryMetadata, _memory),
    ("roofline", RooflineMetadata, _roofline),
    ("timeline", TimelineMetadata, _timeline),
    ("timeline", TimelineSummaryMetadata, _timeline_summary),
)


def _records_of(expr: object, selected: frozenset[type[IRMetadata]]) -> dict[str, object]:
    """Every selected record on *expr*, keyed by the family that owns it."""
    result: dict[str, object] = {}
    for name, metadata_type, project in _RECORDS:
        if metadata_type not in selected:
            continue
        record = get_metadata(expr, metadata_type)
        if record is not None:
            result[name] = project(record, expr)
    return result


@dataclass(frozen=True)
class AnalysisRendering:
    """One annotated program and the report projected from that rendering."""

    data: dict[str, object]
    annotated: str


def selected_types(
    result: AnalysisResult,
) -> tuple[type[IRMetadata], ...]:
    """Return record types requested analyses own and actually wrote.

    Dependencies can produce records nobody requested. Target-selected Analyzer
    ownership, rather than a table, selects the display set; filtering actual
    metadata avoids missing records and keeps report and annotated IR aligned.
    """
    target = result.module.resolve_target()
    owned: set[type[IRMetadata]] = set()
    for analysis in result.analyses:
        owned.update(target.get_analyzer(analysis).produces)
    order: list[type[IRMetadata]] = []
    for metadata_type in result.metadata_types:
        if metadata_type in owned and metadata_type not in order:
            order.append(metadata_type)
    return tuple(order)


def render_analysis(result: AnalysisResult) -> AnalysisRendering:
    """Render one result once for both annotated source and report data."""
    function = result.function
    selected_types_ = selected_types(result)
    selected = frozenset(selected_types_)
    rendered = _render_hir_function(
        function,
        options=PythonPrintOptions(
            show_types=True,
            comment_metadata_types=selected_types_,
        ),
    )
    target = result.module.resolve_target()
    function_records = _records_of(function, selected)
    if "roofline" in result.analyses and "memory" not in function_records:
        support = _roofline_memory_support(result)
        if support is not None:
            function_records["memory"] = support
    data = {
        "target": target.identity,
        "module": result.module.name,
        "function": function.name,
        "topology": result.level,
        "requested": list(result.analyses),
        "executed": list(result.executed),
        "function_records": function_records,
        "calls": _call_records(function, selected, rendered.statements),
    }
    available = set(result.metadata_types)
    if ComputeCostMetadata in selected or (
        "roofline" in data["requested"] and ComputeCostMetadata in available
    ):
        data["totals"] = _work_totals(function)
    return AnalysisRendering(data=data, annotated=rendered.source)


def report(result: AnalysisResult) -> dict[str, object]:
    """Build one report from one composed analysis result."""
    return render_analysis(result).data


def _roofline_memory_support(result: AnalysisResult) -> dict[str, object] | None:
    """The bounded memory fact that explains a requested roofline verdict."""
    record = get_metadata(result.function, MemoryMetadata)
    if record is None:
        return None
    return {
        "footprint": [
            {"level": value.level, "peak_bytes": value.peak_bytes}
            for value in record.footprint
        ]
    }


def _call_records(
    function: Function,
    selected: frozenset[type[IRMetadata]],
    statements: dict[int, _PrintedStatement],
) -> list[dict[str, object]]:
    """Every selected record whose Call has an equation in the rendered view."""
    return [
        {"value": f"{statement.value}:{statement.line}", **records}
        for expr in postorder(function.body)
        if isinstance(expr, Call)
        and (statement := statements.get(id(expr))) is not None
        and (records := _records_of(expr, selected))
    ]


def _work_totals(function: Function) -> dict[str, object]:
    """The multiplicity-aware work recorded on the Function root."""
    record = get_metadata(function, ComputeCostMetadata)
    if record is None:
        return {"flops": {}, "traffic": {}}
    return {
        "flops": dict(record.flops),
        "traffic": _traffic(record.traffic),
    }


def _flop_text(flops: dict[str, int]) -> str:
    return ", ".join(f"{name}={value}" for name, value in sorted(flops.items())) or "0"


def _traffic_text(traffic: dict[str, dict[str, int]]) -> str:
    return (
        ", ".join(
            f"{level}=r{value['read']}/w{value['write']}"
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
        f"function={data['function']} topology={data['topology'] or 'none'}",
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
        lines.extend(f"advisory {note}" for note in memory.get("advisories", ()))
    if "roofline" in records:
        bound = records["roofline"]
        lines.append(
            f"ideal-bound={bound['ideal_ns']}ns by={bound['bound_by']}"
        )
    if "timeline" in records:
        timeline = records["timeline"]
        lines.append(
            f"timeline root={data['module']}::{data['function']} "
            f"local-makespan={timeline['local_makespan_ns']}ns "
            f"waves={timeline['waves']} "
            f"estimated-kernel={timeline['estimated_kernel_ns']}ns"
        )
    return "\n".join(f"# {line}" for line in lines)


def render_json(data: dict[str, object]) -> str:
    """The same report as JSON, sorted so two runs compare byte for byte."""
    return json.dumps(data, indent=2, sort_keys=True)


__all__ = [
    "AnalysisRendering",
    "render_analysis",
    "render_json",
    "render_text",
    "report",
    "selected_types",
]
