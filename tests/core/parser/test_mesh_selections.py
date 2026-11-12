"""A mesh selection given a new layout stays the slice it selects, unless it holds the whole level."""

from __future__ import annotations

from pathlib import Path

from tilefoundry import cli
from tilefoundry.ir import core, types

FIXTURE = Path(__file__).parents[1] / "fixtures" / "parser" / "relabelled_selections.py"


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
