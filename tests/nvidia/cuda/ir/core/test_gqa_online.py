"""Structure of the fixed CUDA online-softmax fixture."""

from __future__ import annotations

from pathlib import Path

from tests.core._corpus import reload
from tilefoundry import cli, inspection
from tilefoundry.ir import core, expr
from tilefoundry_nvidia import cuda


def _walk_ir(node, seen=None):
    if seen is None:
        seen = set()
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    if isinstance(node, expr.Call):
        for arg in node.args:
            yield from _walk_ir(arg, seen)
    elif isinstance(node, expr.Tuple):
        for element in node.elements:
            yield from _walk_ir(element, seen)
    elif isinstance(node, core.LoopRegion):
        for arg in node.args:
            yield from _walk_ir(arg, seen)
        yield from _walk_ir(node.body, seen)
        for value in node.yield_values:
            yield from _walk_ir(value, seen)
    elif isinstance(node, core.MeshRegion):
        yield from _walk_ir(node.body, seen)


def test_static_online_softmax_region_fixed() -> None:
    gqa_static = Path(__file__).parents[2] / "fixtures/logical/gqa_static.py"
    static_online_attend = cli.load_authored_ir(f"{gqa_static}:static_online_attend")
    regions = tuple(
        node
        for node in _walk_ir(static_online_attend.entry_function().body)
        if isinstance(node, core.LoopRegion)
    )
    assert len(regions) == 1
    region = regions[0]
    assert (region.start, region.extent, region.step) == (0, 4096, 1)
    assert {value.name for value in region.params[: len(region.yield_values)]} == {"m", "l", "o"}
    assert static_online_attend.resolve_target() == cuda.Target("nvidia.h200_sxm")
    assert tuple(
        (topology.name, topology.size) for topology in static_online_attend.effective_topologies()
    ) == (("cta", 132),)

    imported = reload(inspection.as_script(static_online_attend), "static_online_attend")
    imported_regions = tuple(
        node
        for node in _walk_ir(imported.entry_function().body)
        if isinstance(node, core.LoopRegion)
    )
    assert len(imported_regions) == 1
    assert imported_regions[0].extent == 4096
