"""Forward relation domain builder — static / dynamic / affine / arity."""
from __future__ import annotations

import isl
import pytest

from tilefoundry.ir.core.expr import Call, Constant
from tilefoundry.ir.types import DType, TensorType
from tilefoundry.ir.types.dim import DimAdd, DimFloorDiv, DimMul, DimVar, simplify_dim
from tilefoundry.visitor_registry.access_relation import AccessRelationResult
from tilefoundry.visitor_registry.relation_build import (
    build_domain,
    shape_from_relation,
    validate_output_map_arity,
)

_I64 = TensorType.scalar(DType.i64)


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


def test_build_domain_affine_extent():
    M = DimVar("M", 1, 4096)
    dom = build_domain((M + 1,))
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.dim(isl.dim_type.SET) == 1


def test_build_domain_rank0():
    dom = build_domain(())
    assert dom.dim(isl.dim_type.SET) == 0


def test_build_domain_floordiv_binds_opaque_param():
    M = DimVar("M", 1, 4096)
    floordiv = simplify_dim(DimFloorDiv, (M, 4))
    dom = build_domain((floordiv,))
    # The dividend's bound [1, 4096) derives the opaque parameter's own
    # bound: M // 4 in [1 // 4, (4096 - 1) // 4 + 1) == [0, 1024).
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.dim(isl.dim_type.SET) == 1
    name = dom.get_dim_name(isl.dim_type.PARAM, 0)
    assert f"0 <= {name} <= 1023" in str(dom)


def test_build_domain_floordiv_same_expr_dedups_param():
    M = DimVar("M", 1, 4096)
    floordiv = simplify_dim(DimFloorDiv, (M, 4))
    # The same canonicalized DimFloorDiv used for two extents in one call
    # binds to a single isl parameter, not two.
    dom = build_domain((floordiv, floordiv))
    assert dom.dim(isl.dim_type.PARAM) == 1


def test_build_domain_floordiv_symbolic_divisor_raises():
    M = DimVar("M", 1, 4096)
    N = DimVar("N", 1, 8)
    floordiv = Call(type=_I64, target=DimFloorDiv(), args=(M, N))
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        build_domain((floordiv,))


def test_build_domain_dimvar_times_const_is_affine():
    M = DimVar("M", 1, 4096)
    mul = Call(type=_I64, target=DimMul(), args=(M, Constant(type=_I64, value=4)))
    dom = build_domain((mul,))
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.dim(isl.dim_type.SET) == 1


def test_build_domain_symbol_times_symbol_raises():
    M = DimVar("M", 1, 4096)
    N = DimVar("N", 1, 4096)
    mul = Call(type=_I64, target=DimMul(), args=(M, N))
    with pytest.raises(NotImplementedError, match="symbolic"):
        build_domain((mul,))


def test_build_domain_same_name_conflicting_bounds_raises():
    with pytest.raises(ValueError, match="conflicting bounds"):
        build_domain((DimVar("S", 1, 8), DimVar("S", 1, 16)))


def test_validate_output_map_arity():
    om = isl.map("{ [m, k, n] -> [m, n] }")
    validate_output_map_arity(om, (1, 1))  # ok
    with pytest.raises(ValueError, match="range rank"):
        validate_output_map_arity(om, (1, 1, 1))


# ─── shape_from_relation ──────────────────────────────────────────────────────


def _ten(shape):
    return TensorType(shape=shape, dtype=DType.f32, layout=None, storage="gmem")


def _relation(extents, out_dst):
    dims = [f"d{i}" for i in range(len(extents))]
    src = "[" + ", ".join(dims) + "]"
    out_map = isl.map(f"{{ {src} -> [{out_dst}] }}")
    return AccessRelationResult(domain=build_domain(extents), maps=(out_map,))


def test_shape_from_relation_static():
    rel = _relation((16, 8), "d0, d1")
    assert shape_from_relation((_ten((16, 8)),), rel) == (16, 8)


def test_shape_from_relation_dimvar_param():
    n = DimVar("N", 1, 64)
    rel = _relation((16, n), "d0, d1")
    # The dynamic axis resolves back to the same DimVar by parameter name.
    assert shape_from_relation((_ten((16, n)),), rel) == (16, n)


def test_shape_from_relation_affine_sum_param():
    n = DimVar("N", 1, 64)
    plus_one = simplify_dim(DimAdd, (1, n))
    rel = _relation((plus_one,), "d0")
    # isl normalizes 1 + n's extent to `n`; the affine inverse rebuilds the
    # nonzero constant offset back onto it (constant term first, matching
    # simplify_dim's own argument order).
    assert shape_from_relation((_ten((plus_one,)),), rel) == (plus_one,)


def test_shape_from_relation_scaled_coefficient_param():
    n = DimVar("N", 1, 64)
    doubled = simplify_dim(DimMul, (2, n))
    rel = _relation((doubled,), "d0")
    assert shape_from_relation((_ten((doubled,)),), rel) == (doubled,)


def test_shape_from_relation_floordiv_param():
    n = DimVar("N", 2048, 1_048_577)
    quarter = simplify_dim(DimFloorDiv, (n, 4))
    rel = _relation((quarter,), "d0")
    # The opaque isl parameter registered for the floordiv resolves back to
    # the original DimExpr through the same lookup a bare DimVar uses.
    assert shape_from_relation((_ten((quarter,)),), rel) == (quarter,)


def test_shape_from_relation_broadcast_constant_axis():
    # A constant output result is a size-1 axis.
    rel = _relation((16, 8), "d0, 0")
    assert shape_from_relation((_ten((16, 8)),), rel) == (16, 1)


def test_shape_from_relation_rank0():
    rel = _relation((), "")
    assert shape_from_relation((_ten(()),), rel) == ()


def test_shape_from_relation_non_projection_fails_closed():
    rel = _relation((16, 8), "d0 + d1")
    with pytest.raises(ValueError, match="pure projection"):
        shape_from_relation((_ten((16, 8)),), rel)
