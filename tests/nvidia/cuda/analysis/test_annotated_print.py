"""The annotated analysis surface, locked against one placed fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.inspection.python_printer import HirPrinter

FIXTURES_ROOT = Path(__file__).parents[1] / "fixtures"
TYPE_FIXTURE = FIXTURES_ROOT / "inspection" / "type_printer_sugar.py"
FIXTURES = (
    (
        TYPE_FIXTURE,
        None,
        TYPE_FIXTURE.with_suffix(".analyzed.txt"),
        ("compute-cost", "memory"),
        False,
    ),
)


def _without_comments(source: str) -> str:
    return "\n".join(line.split("  # ", 1)[0].rstrip() for line in source.splitlines())


@pytest.mark.parametrize(
    ("fixture", "module_name", "golden", "families", "operands"),
    FIXTURES,
)
def test_annotation_keeps_printed_program(
    fixture: Path,
    module_name: str | None,
    golden: Path,
    families: tuple[str, ...],
    operands: bool,
) -> None:
    """Annotation adds metadata to canonical source; it does not restate types.

    The printer fixture's annotated-only golden is shared with its round-trip
    golden, so a type rendered two ways shows up as one diff.
    """
    module = load_authored_ir(f"{fixture}:{module_name}" if module_name else str(fixture))
    result = analyze(*check_program(module, module.entry_function()), analysis=families)
    rendered = render_analysis(result, operands=operands)
    assert rendered.annotated == golden.read_text()
    assert _without_comments(rendered.annotated) == _without_comments(
        HirPrinter().render(result.function, target=result.module.target).source
    )
