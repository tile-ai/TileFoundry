"""Source and text renderings of analysis report data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tilefoundry import analysis, services
from tilefoundry.ir import core, expr

from .analysis_data import render_json, report_data, selected_types
from .python_printer import HirPrinter, PythonPrintOptions
from .values import (
    AdvisorySummary,
    ErrorSummary,
    PerformanceSummaryView,
    Prose,
    ReportIdentity,
    ReportSelection,
)


@dataclass(frozen=True)
class AnalysisRendering:
    """One annotated program and the report projected from that rendering."""

    data: dict[str, object]
    annotated: str
    summary: tuple[expr.IRMetadata, ...] = ()
    printer: HirPrinter | None = None


def render_analysis(result: analysis.Result, *, operands: bool = False) -> AnalysisRendering:
    """Render one result once for both annotated source and report data."""
    selected_types_ = selected_types(result.module, result.analyses, result.metadata_types)
    labels, sources = {}, {}
    printer = HirPrinter()
    for function in reversed(result.functions):
        rendered = printer.render(
            function,
            target=result.module.target,
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
        summary_records(function, {**data, **row}, row["function_records"], selected)
        for function, row in zip(result.functions, data["functions"], strict=True)
    )
    return AnalysisRendering(
        data=data,
        annotated="\n".join((render_summary(summary, printer=printer.definition_printer(result.functions[index])) + "\n\n" if index else "") + source
                            for index, (summary, source) in enumerate(zip(summaries, (sources[id(fn)] for fn in result.functions), strict=True))),
        summary=summaries[0],
        printer=printer.definition_printer(result.function),
    )


def summary_records(
    function: core.Function,
    data: dict[str, object],
    function_records: dict[str, object],
    selected: frozenset[type[expr.IRMetadata]],
) -> tuple[expr.IRMetadata, ...]:
    """One record per summary line: identity, selection, then findings."""
    module_name, _, function_name = data["function"].rpartition(".")
    views: list[expr.IRMetadata] = [
        ReportIdentity(
            target=data["target"],
            module=module_name,
            function=function_name,
            topology=data["topology"] or "none",
        ),
        ReportSelection(requested=tuple(data["requested"]), executed=tuple(data["executed"])),
    ]
    if "totals" in data and "compute-cost" in data["executed"]:
        views.append(expr.get_metadata(function, analysis.ComputeCostMetadata) or analysis.ComputeCostMetadata())
    if "memory" in function_records:
        memory = expr.get_metadata(function, analysis.RegionMemoryMetadata)
        assert memory is not None
        views.append(memory)
        if analysis.RegionMemoryMetadata in selected:
            views.extend(memory_summary(memory)[1:])
    if "roofline" in function_records:
        views.append(expr.get_metadata(function, analysis.RooflineMetadata))
    if "performance" in function_records:
        summary = expr.get_metadata(function, analysis.PerformanceSummaryMetadata)
        views.append(
            PerformanceSummaryView(
                root=f"{module_name}::{function_name}",
                predicted_ns=summary.timeline.end_ns - summary.timeline.start_ns,
                wave="/".join(map(str, summary.wave)),
            )
        )
    return tuple(views)


def report(result: analysis.Result) -> dict[str, object]:
    """Build one report from one composed analysis result."""
    return render_analysis(result).data


def result_json(result: analysis.Result) -> str:
    """Render the report data of one result as JSON, without annotated source."""
    return render_json(
        report_data(
            module=result.module,
            function=result.function,
            analyses=result.analyses,
            topology_level=result.topology_level,
            executed=result.executed,
            metadata_types=result.metadata_types,
            functions=result.functions,
        )
    )


def render_text(rendering: AnalysisRendering) -> str:
    """Render one stable comment line per report conclusion."""
    return render_summary(rendering.summary, printer=rendering.printer)


def memory_summary(memory: analysis.RegionMemoryMetadata) -> tuple[expr.IRMetadata, ...]:
    """Expand the same memory record into summary and diagnostic rows."""
    return (
        memory,
        *memory.reuse_windows,
        *(ErrorSummary(Prose(note)) for note in memory.errors),
        *(AdvisorySummary(Prose(note)) for note in memory.advisories),
    )


def render_summary(records: Iterable[expr.IRMetadata], *, printer, opt_in: frozenset[str] = frozenset()) -> str:
    """Print report rows with the shared metadata comment renderers."""
    nested = (analysis.ReuseWindow, ErrorSummary, AdvisorySummary)
    return "\n".join(
        f"# {'  ' if isinstance(view, nested) else ''}{comment}"
        for view in records
        if (comment := printer.comment(view, opt_in=opt_in)) is not None
    )


services.registry.register(services.report_json, result_json)

__all__ = [
    "AnalysisRendering",
    "render_analysis",
    "render_json",
    "memory_summary",
    "render_summary",
    "render_text",
    "report",
]
