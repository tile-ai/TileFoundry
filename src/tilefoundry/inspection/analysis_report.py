"""Source and text renderings of analysis report data."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.analysis.api import AnalysisResult
from tilefoundry.analysis.metadata import (
    ComputeCostMetadata,
    MemoryMetadata,
    PerformanceSummaryMetadata,
    RooflineMetadata,
    TrafficMetadata,
)
from tilefoundry.analysis.report import _type_text as _type_text
from tilefoundry.analysis.report import (
    render_json,
    report_data,
)
from tilefoundry.analysis.report import (
    selected_types as _selected_types,
)
from tilefoundry.inspection.python_printer import PythonPrintOptions, _render_hir_function
from tilefoundry.inspection.values import (
    AdvisorySummary,
    MemorySummary,
    PerformanceSummaryView,
    Prose,
    ReportIdentity,
    ReportSelection,
    peak_footprint,
    render_comment,
)
from tilefoundry.ir.core import IRMetadata, get_metadata
from tilefoundry.ir.hir.function import Function


@dataclass(frozen=True)
class AnalysisRendering:
    """One annotated program and the report projected from that rendering."""

    data: dict[str, object]
    annotated: str
    summary: tuple[IRMetadata, ...] = ()


def selected_types(result: AnalysisResult) -> tuple[type[IRMetadata], ...]:
    """Return record types requested analyses own and actually wrote."""
    return _selected_types(result.module, result.analyses, result.metadata_types)


def render_analysis(
    result: AnalysisResult, *, operands: bool = False
) -> AnalysisRendering:
    """Render one result once for both annotated source and report data."""
    selected_types_ = selected_types(result)
    rendered = _render_hir_function(
        result.function,
        options=PythonPrintOptions(
            show_types=True,
            comment_metadata_types=selected_types_,
            comment_opt_in=frozenset({"operands"}) if operands else frozenset(),
        ),
    )
    labels = {
        expr_id: f"{statement.value}:{statement.line}"
        for expr_id, statement in rendered.statements.items()
    }
    data = report_data(
        module=result.module,
        function=result.function,
        analyses=result.analyses,
        level=result.level,
        executed=result.executed,
        metadata_types=result.metadata_types,
        call_labels=labels,
    )
    function_records = data["function_records"]
    selected = frozenset(selected_types_)
    return AnalysisRendering(
        data=data,
        annotated=rendered.source,
        summary=_summary(result.function, data, function_records, selected),
    )


def _summary(
    function: Function,
    data: dict[str, object],
    function_records: dict[str, object],
    selected: frozenset[type[IRMetadata]],
) -> tuple[IRMetadata, ...]:
    """One record per summary line: identity, selection, then findings."""
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
    if "totals" in data and "compute-cost" in data["executed"]:
        views.append(get_metadata(function, ComputeCostMetadata) or ComputeCostMetadata())
    if "traffic" in function_records:
        views.append(get_metadata(function, TrafficMetadata) or TrafficMetadata())
    if "memory" in function_records:
        memory = get_metadata(function, MemoryMetadata)
        views.append(MemorySummary(peak_footprint(memory)))
        if MemoryMetadata in selected:
            views.extend(AdvisorySummary(Prose(note)) for note in memory.advisories)
    if "roofline" in function_records:
        views.append(get_metadata(function, RooflineMetadata))
    if "performance" in function_records:
        summary = get_metadata(function, PerformanceSummaryMetadata)
        views.append(
            PerformanceSummaryView(
                root=f"{data['module']}::{data['function']}",
                predicted_ns=summary.timeline.end_ns - summary.timeline.start_ns,
                waves=summary.waves,
            )
        )
    return tuple(views)


def report(result: AnalysisResult) -> dict[str, object]:
    """Build one report from one composed analysis result."""
    return render_analysis(result).data


def render_text(rendering: AnalysisRendering) -> str:
    """Render one stable comment line per report conclusion."""
    return "\n".join(f"# {render_comment(view)}" for view in rendering.summary)


__all__ = [
    "AnalysisRendering",
    "render_analysis",
    "render_json",
    "render_text",
    "report",
    "selected_types",
]
