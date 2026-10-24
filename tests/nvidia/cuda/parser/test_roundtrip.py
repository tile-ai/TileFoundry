"""Parser-owned checks for canonical grid-loop bindings."""


from tests.core._source import import_dsl
from tests.nvidia.cuda.fixtures.placed.region_boundaries import RegionBoundaries
from tilefoundry.inspection import as_script
from tilefoundry.ir.core import binding_name
from tilefoundry.ir.hir.mesh_region import MeshRegion
from tilefoundry.ir.visitor import collect_exprs, expr_children


def test_region_boundaries_round_trip() -> None:
    """Printing and reparsing preserves nested scopes and escaped values."""
    once = as_script(RegionBoundaries)
    restored = import_dsl(once, "RegionBoundaries")
    canonical = as_script(restored)
    assert as_script(import_dsl(canonical, "RegionBoundaries")) == canonical

    body = restored.entry_function().body
    producer = next(
        expr
        for expr in collect_exprs(body)
        if isinstance(expr, MeshRegion) and binding_name(expr.body) == "v2"
    )
    assert sum(producer in expr_children(expr) for expr in collect_exprs(body)) == 2
