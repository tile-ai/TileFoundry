"""DOT rendering retains HIR loop regions as opaque leaves."""

from __future__ import annotations

from tests.nvidia.cuda.fixtures.logical.gqa_static import static_online_attend
from tilefoundry.dsl import *
from tilefoundry.inspection.dot import hir_function_to_dot


def test_hir_function_dot_keeps_loop_regions_as_opaque_leaves() -> None:
    """The public DOT form keeps structured regions as white leaf boxes."""
    dot = hir_function_to_dot(static_online_attend.entry_function())

    assert len(dot.splitlines()) == 33
    assert 'label="LoopRegion", fillcolor="#ffffff"' in dot
    assert "TupleType(" not in dot
