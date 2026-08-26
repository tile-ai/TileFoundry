"""isl_utility — dim_range, to_domain encode, to_dim decode."""

from __future__ import annotations

import isl
import pytest

from tilefoundry.ir.core.expr import Call, Var
from tilefoundry.ir.types import TensorType
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
from tilefoundry.ir.types.dim_isl import dim_range, normalize_dim, to_dim, to_domain

P = DimVar("P", 2048, 1_048_577)
Q = DimVar("Q", 2, 33)


def test_normalize_dim_uses_isl_affine_normal_form():
    verbose = simplify_dim(
        DimFloorDiv,
        (
            simplify_dim(
                DimAdd,
                (simplify_dim(DimSub, (simplify_dim(DimAdd, (P, 0)), 0)), 0),
            ),
            1,
        ),
    )
    quotient = simplify_dim(DimFloorDiv, (simplify_dim(DimAdd, (P, 8)), 4))
    expected = simplify_dim(
        DimAdd,
        (simplify_dim(DimFloorDiv, (P, 4)), 2),
    )

    constant = simplify_dim(DimMul, (simplify_dim(DimAdd, (4, 2)), 3))

    assert normalize_dim(constant) == 18
    assert isinstance(normalize_dim(constant), int)
    assert normalize_dim(verbose) is P
    assert normalize_dim(quotient) == expected


def test_normalize_dim_leaves_unsupported_expressions_unchanged():
    symbolic_divisor = simplify_dim(
        DimFloorDiv,
        (simplify_dim(DimMul, (P, Q)), DimVar("G", 1, 65)),
    )
    piecewise = simplify_dim(DimMin, (P, 8192))

    assert normalize_dim(symbolic_divisor) is symbolic_divisor
    assert normalize_dim(piecewise) is piecewise


def test_normalize_dim_keys_runtime_parameters_by_object_identity():
    scalar = TensorType.umat_scalar()
    first = Var(type=scalar, name="start")
    second = Var(type=scalar, name="start")

    def distance(left, right):
        return simplify_dim(
            DimSub,
            (
                simplify_dim(DimAdd, (left, 9)),
                simplify_dim(DimAdd, (right, 1)),
            ),
        )

    assert normalize_dim(distance(first, first)) == 8
    distinct = normalize_dim(distance(first, second))
    assert isinstance(distinct, Call)

    def vars_in(value):
        if isinstance(value, Var):
            return [value]
        if isinstance(value, Call):
            return [leaf for arg in value.args for leaf in vars_in(arg)]
        return []

    leaves = vars_in(distinct)
    assert any(leaf is first for leaf in leaves)
    assert any(leaf is second for leaf in leaves)


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
    """Static extents inline.

    Static extents inline; a bare DimVar keeps its own param name; a
    composite mints one opaque param bounded by ``dim_range`` and dedups
    across axes on the canonical expression.
    """
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


def test_round_trip_lossless_for_every_dim_kind():
    """Encode every kind of ``ShapeDim`` into a domain and read it back out.

    The decode side is asserted here rather than on its own, because what matters
    is that it is the exact inverse: a constant comes back an ``int``, a parameter
    comes back the very same ``DimVar`` object, and a parameter with no entry in
    the map is refused instead of being invented as an opaque dim nothing can
    resolve later.
    """
    assert to_dim(isl.pw_aff("{ [42] }"), {}) == 42
    named = isl.pw_aff("[P] -> { [P] }")
    assert to_dim(named, {"P": P}) is P
    with pytest.raises(ValueError, match="no known ShapeDim"):
        to_dim(named, {})

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
