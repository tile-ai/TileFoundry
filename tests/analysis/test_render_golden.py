"""The annotated analysis surface, locked against one placed fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tilefoundry.analysis import analyze
from tilefoundry.inspection import as_script
from tilefoundry.inspection.analysis_report import render_analysis
from tilefoundry.ir.core.module import Module

FIXTURE = Path(__file__).parents[1] / "fixtures" / "inspection" / "type_printer_sugar.py"


def _module_in(path: Path) -> Module:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return next(value for value in vars(loaded).values() if isinstance(value, Module))


def _without_comments(source: str) -> str:
    return "\n".join(line.split("  # ", 1)[0].rstrip() for line in source.splitlines())


def test_analysis_annotates_the_printed_program_without_changing_it() -> None:
    """Annotation adds metadata to canonical source; it does not restate types.

    The golden shares its fixture with the round-trip golden, so a type that
    reads one way in emitted code and another in an annotation shows up here as
    a diff rather than as two goldens that drifted apart.
    """
    module = _module_in(FIXTURE)
    result = analyze(module, module.entry_function(), analysis=("compute-cost", "memory"))
    annotated = render_analysis(result).annotated

    assert annotated == FIXTURE.with_suffix(".analyzed.txt").read_text()
    assert _without_comments(annotated) == _without_comments(as_script(result.function))
