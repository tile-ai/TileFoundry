"""Cover online GQA specialization and CTA distribution strategies.

Each variant and its dispatch prototype must match the torch decode reference.
A nonaligned context length fails closed instead of dropping a split tail.

See [hir §2](docs/spec/hir.md#2-function-specialization-api).
"""

from __future__ import annotations

from tests.core._source import import_dsl
from tests.nvidia.cuda.fixtures.logical.gqa_static import static_online_attend
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import Call, Tuple
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry_nvidia import CudaTarget


def _walk_ir(expr, seen=None):
    if seen is None:
        seen = set()
    if expr is None or id(expr) in seen:
        return
    seen.add(id(expr))
    yield expr
    if isinstance(expr, Call):
        for arg in expr.args:
            yield from _walk_ir(arg, seen)
    elif isinstance(expr, Tuple):
        for element in expr.elements:
            yield from _walk_ir(element, seen)
    elif isinstance(expr, LoopRegion):
        for arg in expr.args:
            yield from _walk_ir(arg, seen)
        yield from _walk_ir(expr.body, seen)
        for value in expr.yield_values:
            yield from _walk_ir(value, seen)
    elif isinstance(expr, MeshRegion):
        yield from _walk_ir(expr.body, seen)


def test_static_fixture_has_one_fixed_online_softmax_region() -> None:
    regions = tuple(
        expr
        for expr in _walk_ir(static_online_attend.entry_function().body)
        if isinstance(expr, LoopRegion)
    )
    assert len(regions) == 1
    region = regions[0]
    assert (region.start, region.extent, region.step) == (0, 4096, 1)
    assert {value.name for value in region.params[: len(region.yield_values)]} == {"m", "l", "o"}
    assert static_online_attend.resolve_target() == CudaTarget("nvidia.h200_sxm")
    assert tuple(
        (topology.name, topology.size) for topology in static_online_attend.effective_topologies()
    ) == (("cta", 132),)

    imported = import_dsl(as_script(static_online_attend))
    imported_regions = tuple(
        expr
        for expr in _walk_ir(imported.entry_function().body)
        if isinstance(expr, LoopRegion)
    )
    assert len(imported_regions) == 1
    assert imported_regions[0].extent == 4096
