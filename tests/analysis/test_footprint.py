"""Focused contracts for footprint and reuse-window helpers."""

from tilefoundry.analysis.footprint import axis_label
from tilefoundry.ir.types.shard import Mesh


def test_an_unnamed_mesh_axis_uses_its_position() -> None:
    mesh = Mesh(("cta",), layout=(2, 3), names=("row",))

    assert axis_label(mesh, 0) == "cta.row"
    assert axis_label(mesh, 1) == "cta[1]"
