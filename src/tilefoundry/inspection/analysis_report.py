"""Source and text renderings of analysis report data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tilefoundry.analysis.api import AnalysisResult
from tilefoundry.analysis.metadata import (
    ComputeCostMetadata,
    PerformanceSummaryMetadata,
    RegionMemoryMetadata,
    ReuseWindow,
    RooflineMetadata,
)
from tilefoundry.analysis.report import _type_text as _type_text
from tilefoundry.analysis.report import (
    render_json,
    report_data,
)
from tilefoundry.analysis.report import (
    selected_types as _selected_types,
)
from tilefoundry.inspection.python_printer import HirPrinter, PythonPrintOptions
from tilefoundry.inspection.values import (
    AdvisorySummary,
    ErrorSummary,
    PerformanceSummaryView,
    Prose,
    ReportIdentity,
    ReportSelection,
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
    printer: HirPrinter | None = None


def selected_types(result: AnalysisResult) -> tuple[type[IRMetadata], ...]:
    """Return record types requested analyses own and actually wrote."""
    return _selected_types(result.module, result.analyses, result.metadata_types)


def render_analysis(result: AnalysisResult, *, operands: bool = False) -> AnalysisRendering:
    """Render one result once for both annotated source and report data."""
    selected_types_ = selected_types(result)
    labels, sources = {}, {}
    printer = HirPrinter()
    for function in reversed(result.functions):
        rendered = printer.render(
            function,
            options=PythonPrintOptions(
                show_types=True, comment_metadata_types=selected_types_,
                comment_opt_in=frozenset({"operands"}) if operands else frozenset(),
            ),
        )
        sources[id(function)] = rendered.source
        labels.update({identity: f"{statement.value}:{statement.line}"
                       for identity, statement in rendered.statements.items()})
    data = report_data(
        module=result.module, function=result.function, analyses=result.analyses,
        topology_level=result.topology_level, executed=result.executed,
        metadata_types=result.metadata_types, labels=labels, functions=result.functions, printer=printer,
    )
    selected = frozenset(selected_types_)
    summaries = tuple(
        _summary(function, {**data, **row}, row["function_records"], selected)
        for function, row in zip(result.functions, data["functions"], strict=True)
    )
    return AnalysisRendering(
        data=data,
        annotated="\n".join((render_summary(summary, printer=printer._definitions[id(result.functions[index])]) + "\n\n" if index else "") + source
                            for index, (summary, source) in enumerate(zip(summaries, (sources[id(fn)] for fn in result.functions), strict=True))),
        summary=summaries[0],
        printer=printer._definitions[id(result.function)],
    )


def _summary(
    function: Function,
    data: dict[str, object],
    function_records: dict[str, object],
    selected: frozenset[type[IRMetadata]],
) -> tuple[IRMetadata, ...]:
    """One record per summary line: identity, selection, then findings."""
    module_name, _, function_name = data["function"].rpartition(".")
    wave = data["wave"]
    counted = wave["counted"]
    declared = wave["declared"]
    views: list[IRMetadata] = [
        ReportIdentity(
            target=data["target"],
            module=module_name,
            function=function_name,
            topology=data["topology"] or "none",
            wave=f"{counted}/{declared}" if counted and declared else "",
        ),
        ReportSelection(requested=tuple(data["requested"]), executed=tuple(data["executed"])),
    ]
    if "totals" in data and "compute-cost" in data["executed"]:
        views.append(get_metadata(function, ComputeCostMetadata) or ComputeCostMetadata())
    if "memory" in function_records:
        memory = get_metadata(function, RegionMemoryMetadata)
        assert memory is not None
        views.append(memory)
        if RegionMemoryMetadata in selected:
            views.extend(memory_summary(memory)[1:])
    if "roofline" in function_records:
        views.append(get_metadata(function, RooflineMetadata))
    if "performance" in function_records:
        summary = get_metadata(function, PerformanceSummaryMetadata)
        views.append(
            PerformanceSummaryView(
                root=f"{module_name}::{function_name}",
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
    return render_summary(rendering.summary, printer=rendering.printer)


def memory_summary(memory: RegionMemoryMetadata) -> tuple[IRMetadata, ...]:
    """Expand the same memory record into summary and diagnostic rows."""
    return (
        memory,
        *memory.reuse_windows,
        *(ErrorSummary(Prose(note)) for note in memory.errors),
        *(AdvisorySummary(Prose(note)) for note in memory.advisories),
    )


def render_summary(records: Iterable[IRMetadata], *, opt_in: frozenset[str] = frozenset(), printer=None) -> str:
    """Print report rows with the shared metadata comment renderers."""
    nested = (ReuseWindow, ErrorSummary, AdvisorySummary)
    return "\n".join(
        f"# {'  ' if isinstance(view, nested) else ''}{comment}"
        for view in records
        if (comment := render_comment(view, opt_in=opt_in, printer=printer)) is not None
    )


__all__ = [
    "AnalysisRendering",
    "render_analysis",
    "render_json",
    "memory_summary",
    "render_summary",
    "render_text",
    "report",
    "selected_types",
]
