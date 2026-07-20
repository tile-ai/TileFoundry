"""isl_utility — dim_range, to_domain encode, to_dim decode."""
from __future__ import annotations

import isl
import pytest

from tilefoundry.ir.types.dim import (
    DimAdd,
    DimFloorDiv,
    DimMax,
    DimMin,
    DimMod,
    DimMul,
    DimSub,
    DimVar,
    simplify_dim,
)
from tilefoundry.visitor_registry.isl_utility import dim_range, to_dim, to_domain

P = DimVar("P", 2048, 1_048_577)
Q = DimVar("Q", 2, 33)


def test_dim_range_interval_arithmetic():
    """Conservative half-open interval per dim kind, incl. nesting."""
    assert dim_range(7) == (7, 8)
    assert dim_range(P) == (P.lo, P.hi)
    assert dim_range(simplify_dim(DimAdd, (128, P))) == (128 + P.lo, 128 + P.hi)
    assert dim_range(simplify_dim(DimSub, (P, Q))) == (P.lo - (Q.hi - 1), P.hi - Q.lo)
    assert dim_range(simplify_dim(DimMul, (4, P))) == (4 * P.lo, 4 * (P.hi - 1) + 1)
    assert dim_range(simplify_dim(DimMul, (P, Q))) == (
        P.lo * Q.lo,
        (P.hi - 1) * (Q.hi - 1) + 1,
    )
    assert dim_range(simplify_dim(DimFloorDiv, (P, 4))) == (P.lo // 4, (P.hi - 1) // 4 + 1)
    assert dim_range(simplify_dim(DimMod, (P, 128))) == (0, 128)
    assert dim_range(simplify_dim(DimMax, (P, Q))) == (max(P.lo, Q.lo), max(P.hi, Q.hi))
    assert dim_range(simplify_dim(DimMin, (P, Q))) == (min(P.lo, Q.lo), min(P.hi, Q.hi))

    inner = simplify_dim(DimFloorDiv, (P, 4))
    outer = simplify_dim(DimFloorDiv, (inner, 2))
    ilo, ihi = dim_range(inner)
    assert dim_range(outer) == (ilo // 2, (ihi - 1) // 2 + 1)


def test_dim_range_symbolic_divisor_unsupported():
    n = DimVar("N", 1, 8)
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        dim_range(simplify_dim(DimFloorDiv, (P, n)))
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        dim_range(simplify_dim(DimMod, (P, n)))


def test_to_domain_encoding():
    """Static extents inline; a bare DimVar keeps its own param name; a
    composite mints one opaque param bounded by ``dim_range`` and dedups
    across axes on the canonical expression."""
    dom, param_map = to_domain((8, 4))
    assert dom.dim(isl.dim_type.PARAM) == 0
    assert dom.dim(isl.dim_type.SET) == 2
    assert param_map == {}

    dom, param_map = to_domain((P,))
    assert dom.get_dim_name(isl.dim_type.PARAM, 0) == "P"
    assert param_map == {"P": P}

    d = simplify_dim(DimFloorDiv, (P, 4))
    dom, param_map = to_domain((d, 128, d))
    assert dom.dim(isl.dim_type.PARAM) == 1
    name = dom.get_dim_name(isl.dim_type.PARAM, 0)
    lo, hi = dim_range(d)
    assert f"{lo} <= {name} <= {hi - 1}" in str(dom)
    assert param_map[name] is d

    dom, param_map = to_domain(())
    assert dom.dim(isl.dim_type.SET) == 0
    assert param_map == {}


def test_to_domain_same_name_conflicting_bounds_raises():
    with pytest.raises(ValueError, match="conflicting bounds"):
        to_domain((DimVar("S", 1, 8), DimVar("S", 1, 16)))


def test_to_dim_decode():
    assert to_dim(isl.pw_aff("{ [42] }"), {}) == 42
    pa = isl.pw_aff("[P] -> { [P] }")
    assert to_dim(pa, {"P": P}) is P
    with pytest.raises(ValueError, match="no known ShapeDim"):
        to_dim(pa, {})


def test_round_trip_lossless_for_every_dim_kind():
    dims = (
        128,
        P,
        simplify_dim(DimAdd, (128, P)),
        simplify_dim(DimSub, (P, 3)),
        simplify_dim(DimFloorDiv, (P, 4)),
        simplify_dim(DimFloorDiv, (simplify_dim(DimFloorDiv, (P, 4)), 2)),
        simplify_dim(DimMul, (P, Q)),
        simplify_dim(DimMod, (P, 128)),
        simplify_dim(DimAdd, (128, simplify_dim(DimFloorDiv, (P, 4)))),
    )
    domain, param_map = to_domain(dims)
    recovered = tuple(
        to_dim(domain.dim_max(i).add_constant(1), param_map) for i in range(len(dims))
    )
    assert recovered == dims
