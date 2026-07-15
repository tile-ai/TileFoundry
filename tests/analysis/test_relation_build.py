"""relation_build — thin isl_utility consumer: build_domain, shape_from_relation, arity."""
from __future__ import annotations

import isl
import pytest

from tilefoundry.ir.types.dim import DimFloorDiv, DimVar, simplify_dim
from tilefoundry.visitor_registry.isl_utility import to_domain
from tilefoundry.visitor_registry.access_relation import AccessRelationResult
from tilefoundry.visitor_registry.relation_build import (
    build_domain,
    shape_from_relation,
    validate_output_map_arity,
)


def test_build_domain_static_constant_constraints():
    dom = build_domain((8, 4))
    # Static extents become constant upper bounds; no isl parameters.
    assert dom.dim(isl.dim_type.PARAM) == 0
    assert dom.dim(isl.dim_type.SET) == 2
    # The whole [0,8)x[0,4) box is 32 points.
    assert int(dom.count_val().num_si()) == 32


def test_build_domain_dynamic_dimvar_is_param():
    M = DimVar("M", 1, 4096)
    dom = build_domain((M, 128))
    # The dynamic extent becomes one isl parameter carrying its bound.
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.dim(isl.dim_type.SET) == 2


def test_build_domain_composite_extent_binds_one_opaque_param():
    M = DimVar("M", 1, 4096)
    dom = build_domain((simplify_dim(DimFloorDiv, (M, 4)),))
    # build_domain is a thin to_domain(extents).domain wrapper; dim_range /
    # opaque-param mechanics are isl_utility's own test surface.
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.dim(isl.dim_type.SET) == 1


def test_build_domain_rank0():
    dom = build_domain(())
    assert dom.dim(isl.dim_type.SET) == 0


def test_build_domain_same_name_conflicting_bounds_raises():
    with pytest.raises(ValueError, match="conflicting bounds"):
        build_domain((DimVar("S", 1, 8), DimVar("S", 1, 16)))


def test_validate_output_map_arity():
    om = isl.map("{ [m, k, n] -> [m, n] }")
    validate_output_map_arity(om, (1, 1))  # ok
    with pytest.raises(ValueError, match="range rank"):
        validate_output_map_arity(om, (1, 1, 1))


# ─── shape_from_relation ──────────────────────────────────────────────────────


def _relation(extents, out_dst):
    domain, param_map = to_domain(extents)
    dims = [f"d{i}" for i in range(len(extents))]
    src = "[" + ", ".join(dims) + "]"
    out_map = isl.map(f"{{ {src} -> [{out_dst}] }}")
    return AccessRelationResult(domain=domain, maps=(out_map,), param_map=param_map)


def test_shape_from_relation_static():
    rel = _relation((16, 8), "d0, d1")
    assert shape_from_relation(rel) == (16, 8)


def test_shape_from_relation_dimvar_param():
    n = DimVar("N", 1, 64)
    rel = _relation((16, n), "d0, d1")
    # The dynamic axis resolves back to the same DimVar by parameter name.
    assert shape_from_relation(rel) == (16, n)


def test_shape_from_relation_composite_param():
    n = DimVar("N", 2048, 1_048_577)
    quarter = simplify_dim(DimFloorDiv, (n, 4))
    rel = _relation((quarter,), "d0")
    # The opaque parameter minted for the composite expr resolves back to
    # the original DimExpr via relation.param_map.
    assert shape_from_relation(rel) == (quarter,)


def test_shape_from_relation_broadcast_constant_axis():
    # A constant output result is a size-1 axis.
    rel = _relation((16, 8), "d0, 0")
    assert shape_from_relation(rel) == (16, 1)


def test_shape_from_relation_rank0():
    rel = _relation((), "")
    assert shape_from_relation(rel) == ()


def test_shape_from_relation_non_projection_fails_closed():
    rel = _relation((16, 8), "d0 + d1")
    with pytest.raises(ValueError, match="pure projection"):
        shape_from_relation(rel)
