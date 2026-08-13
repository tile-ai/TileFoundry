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
from dataclasses import asdict, dataclass

from tilefoundry.analysis import (
    ComputeCostMetadata,
    MemoryMetadata,
    RooflineMetadata,
    TimelineSummaryMetadata,
)
from tilefoundry.analysis.api import AnalysisResult
from tilefoundry.analysis.walk import postorder, tensor_types
from tilefoundry.inspection.python_printer import (
    PythonPrintOptions,
    _PrintedStatement,
    _render_hir_function,
)
from tilefoundry.inspection.values import (
    AdvisorySummary,
    MemorySummary,
    Prose,
    ReportIdentity,
    ReportSelection,
    TimelineSummaryView,
    declared_records,
    expr_field,
    family_of,
    peak_footprint,
    render_comment,
    render_record,
)
from tilefoundry.ir.core import Call, IRMetadata, binding_name, get_metadata
from tilefoundry.ir.hir.function import Function


def _type_text(type_: object) -> str:
    """One operand's type, short enough to sit on a report line."""
    tensors = tensor_types(type_)
    if not tensors:
        return str(type_)
    first = tensors[0]
    shape = ",".join(str(dim) for dim in first.shape)
    more = "+" if len(tensors) > 1 else ""
    storage = f" {first.storage}"
    return f"{first.dtype.name}[{shape}]{more}{storage}"


def _operand_name(operand: object) -> str:
    """What to call this operand, from the program rather than from the record."""
    name = binding_name(operand) or getattr(operand, "name", None)
    if name:
        return name
    if isinstance(operand, Call):
        return type(operand.target).__name__.lower()
    return type(operand).__name__.lower()


def _operands(
    record: ComputeCostMetadata, expr: object
) -> list[dict[str, object]] | None:
    """Each recorded amount, against the operand of the program it was charged to.

    ``None`` where there is no such split to report: a Function Call charges its
    callee's total, and an operand position names nothing there.
    """
    if not isinstance(expr, Call) or not record.operands:
        return None
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


expr_field(ComputeCostMetadata, "operands", _operands)


def _records_of(expr: object, selected: frozenset[type[IRMetadata]]) -> dict[str, object]:
    """Every selected record on *expr*, keyed by the family that owns it."""
    result: dict[str, object] = {}
    for metadata_type in declared_records():
        if metadata_type not in selected:
            continue
        record = get_metadata(expr, metadata_type)
        if record is not None:
            result[family_of(metadata_type)] = render_record(record, expr)
    return result


@dataclass(frozen=True)
class AnalysisRendering:
    """One annotated program and the report projected from that rendering.

    *summary* is one record per summary line, in reading order. They are the same
    records the equations are annotated from, so a summary line and an equation
    cannot state one number two ways.
    """

    data: dict[str, object]
    annotated: str
    summary: tuple[IRMetadata, ...] = ()


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


def render_analysis(
    result: AnalysisResult, *, operands: bool = False
) -> AnalysisRendering:
    """Render one result once for both annotated source and report data.

    *operands* asks the annotation for the per-operand split of the traffic it
    already states. JSON carries it either way: it is read by programs, and a
    reader of the text is not looking that closely by default.
    """
    function = result.function
    selected_types_ = selected_types(result)
    selected = frozenset(selected_types_)
    rendered = _render_hir_function(
        function,
        options=PythonPrintOptions(
            show_types=True,
            comment_metadata_types=selected_types_,
            comment_opt_in=frozenset({"operands"}) if operands else frozenset(),
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
    return AnalysisRendering(
        data=data,
        annotated=rendered.source,
        summary=_summary(function, data, function_records, selected),
    )


def _summary(
    function: Function,
    data: dict[str, object],
    function_records: dict[str, object],
    selected: frozenset[type[IRMetadata]],
) -> tuple[IRMetadata, ...]:
    """One record per summary line: what this report is about, then its findings.

    A conclusion is stated by the record that holds it, on the same terms the
    equations state theirs. Which lines appear is the same question as which
    records the report carries, so it is asked of those and not of the text.
    """
    views: list[IRMetadata] = [
        ReportIdentity(
            target=data["target"],
            module=data["module"],
            function=data["function"],
            topology=data["topology"] or "none",
        ),
        ReportSelection(
            requested=tuple(data["requested"]), executed=tuple(data["executed"])
        ),
    ]
    if "totals" in data:
        views.append(get_metadata(function, ComputeCostMetadata) or ComputeCostMetadata())
    if "memory" in function_records:
        memory = get_metadata(function, MemoryMetadata)
        views.append(MemorySummary(peak_footprint(memory)))
        if MemoryMetadata in selected:
            views.extend(AdvisorySummary(Prose(note)) for note in memory.advisories)
    if "roofline" in function_records:
        views.append(get_metadata(function, RooflineMetadata))
    if "timeline" in function_records:
        summary = get_metadata(function, TimelineSummaryMetadata)
        views.append(
            TimelineSummaryView(
                root=f"{data['module']}::{data['function']}", **asdict(summary)
            )
        )
    return tuple(views)


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
    reported = render_record(record, function)
    return {"flops": reported["flops"], "traffic": reported["traffic"]}


def render_text(rendering: AnalysisRendering) -> str:
    """The report as one stable line per conclusion, each prefixed with ``#``.

    Every line is one record walked the way an annotated equation is, so nothing
    here knows what a family has: a conclusion appears when a record states it.
    """
    return "\n".join(f"# {render_comment(view)}" for view in rendering.summary)


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
