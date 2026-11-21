"""A relabelled mesh selection stays the slice it selects, unless it holds the whole level."""

from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry import cli, parser
from tilefoundry.ir import core, types

FIXTURE = Path(__file__).parents[1] / "fixtures" / "parser" / "relabelled_selections.py"
DIAGNOSTICS = Path(__file__).parents[1] / "fixtures" / "diagnostics"


def test_a_relabelled_selection_stays_a_slice_unless_it_holds_its_level() -> None:
    function = cli.load_authored_ir(str(FIXTURE)).entry_function()
    meshes = {
        region.mesh.names: region.mesh
        for region in core.collect_exprs(function.body)
        if isinstance(region, core.MeshRegion)
    }
    head, tail, grid = meshes[("head",)], meshes[("tail",)], meshes[("row", "column")]

    assert isinstance(head.layout, types.ComposedLayout) and head.layout.offset == 0
    assert isinstance(tail.layout, types.ComposedLayout) and tail.layout.offset == 16
    assert not isinstance(grid.layout, types.ComposedLayout)


@pytest.mark.parametrize(
    ("fixture", "line"),
    (("slice_two_unit_sets.py", 14), ("slice_of_a_slice.py", 15), ("slice_outside_scope.py", 15)),
)
def test_a_slice_in_a_type_is_refused_at_its_line(fixture, line) -> None:
    """Two unit sets in one layout, a slice of a slice, units outside the scope."""
    with pytest.raises(parser.ParseError) as refused:
        cli.load_authored_ir(str(DIAGNOSTICS / fixture))
    assert refused.value.node.lineno == line
