"""DOT rendering retains HIR loop regions as opaque leaves."""

from __future__ import annotations

from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.inspection.dot import hir_function_to_dot


def test_dot_keeps_loop_regions_opaque() -> None:
    """The public DOT form keeps structured regions as white leaf boxes."""
    gqa_static = Path(__file__).parents[1] / "fixtures/logical/gqa_static.py"
    static_online_attend = load_authored_ir(f"{gqa_static}:static_online_attend")
    dot = hir_function_to_dot(static_online_attend.entry_function())

    assert len(dot.splitlines()) == 33
    assert 'label="LoopRegion", fillcolor="#ffffff"' in dot
    assert "TupleType(" not in dot
