"""User-visible provenance for destructured tuple projections."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.fixtures.diagnostics.tuple_projection_diagnostic import TupleProjectionDiagnostic
from tilefoundry.analysis.api import analyze
from tilefoundry.analysis.errors import AnalysisError

_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "diagnostics"
    / "tuple_projection_diagnostic.py"
)


def _values_target_location() -> tuple[int, int]:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Tuple)
        and tuple(target.id for target in node.targets[0].elts if isinstance(target, ast.Name))
        == ("values", "indices")
    )
    assert isinstance(assignment.targets[0], ast.Tuple)
    values = assignment.targets[0].elts[0]
    assert isinstance(values, ast.Name)
    return values.lineno, values.col_offset + 1


def test_performance_points_at_the_tuple_target_when_its_projection_is_invalid() -> None:
    line, column = _values_target_location()

    with pytest.raises(AnalysisError) as exc_info:
        analyze(
            TupleProjectionDiagnostic,
            TupleProjectionDiagnostic.entry_function(),
            analysis="performance",
        )

    message = str(exc_info.value)
    assert f"{_SOURCE}:{line}:{column}" in message
    assert "op=TupleGetItem" in message
    assert "has no cta execution domain" in message
