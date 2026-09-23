"""The annotated analysis surface, locked against one placed fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.inspection import as_script
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.ir.core.module import Module

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


def _module_in(path: Path, name: str | None) -> Module:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    if name is not None:
        module = getattr(loaded, name)
        assert isinstance(module, Module)
        return module
    return next(value for value in vars(loaded).values() if isinstance(value, Module))


def _without_comments(source: str) -> str:
    return "\n".join(line.split("  # ", 1)[0].rstrip() for line in source.splitlines())


@pytest.mark.parametrize(
    ("fixture", "module_name", "golden", "families", "operands"),
    FIXTURES,
)
def test_analysis_annotates_the_printed_program_without_changing_it(
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
    module = _module_in(fixture, module_name)
    result = analyze(module, module.entry_function(), analysis=families)
    rendered = render_analysis(result, operands=operands)
    assert rendered.annotated == golden.read_text()
    assert _without_comments(rendered.annotated) == _without_comments(
        as_script(result.function)
    )
