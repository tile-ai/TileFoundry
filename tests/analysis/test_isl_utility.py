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


# ─── dim_range ─────────────────────────────────────────────────────────────


def test_dim_range_const_and_var():
    assert dim_range(7) == (7, 8)
    assert dim_range(P) == (P.lo, P.hi)


def test_dim_range_add_sub():
    assert dim_range(simplify_dim(DimAdd, (128, P))) == (128 + P.lo, 128 + P.hi)
    assert dim_range(simplify_dim(DimSub, (P, Q))) == (P.lo - (Q.hi - 1), P.hi - Q.lo)


def test_dim_range_mul_const_and_symbolic():
    assert dim_range(simplify_dim(DimMul, (4, P))) == (4 * P.lo, 4 * (P.hi - 1) + 1)
    lo, hi = dim_range(simplify_dim(DimMul, (P, Q)))
    assert (lo, hi) == (P.lo * Q.lo, (P.hi - 1) * (Q.hi - 1) + 1)


def test_dim_range_floordiv_const():
    assert dim_range(simplify_dim(DimFloorDiv, (P, 4))) == (P.lo // 4, (P.hi - 1) // 4 + 1)


def test_dim_range_floordiv_symbolic_divisor_raises():
    n = DimVar("N", 1, 8)
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        dim_range(simplify_dim(DimFloorDiv, (P, n)))


def test_dim_range_mod_const():
    assert dim_range(simplify_dim(DimMod, (P, 128))) == (0, 128)


def test_dim_range_mod_symbolic_divisor_raises():
    n = DimVar("N", 1, 8)
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        dim_range(simplify_dim(DimMod, (P, n)))


def test_dim_range_max_min():
    assert dim_range(simplify_dim(DimMax, (P, Q))) == (max(P.lo, Q.lo), max(P.hi, Q.hi))
    assert dim_range(simplify_dim(DimMin, (P, Q))) == (min(P.lo, Q.lo), min(P.hi, Q.hi))


def test_dim_range_nested_floordiv():
    inner = simplify_dim(DimFloorDiv, (P, 4))
    outer = simplify_dim(DimFloorDiv, (inner, 2))
    ilo, ihi = dim_range(inner)
    assert dim_range(outer) == (ilo // 2, (ihi - 1) // 2 + 1)


# ─── to_domain ──────────────────────────────────────────────────────────────


def test_to_domain_static_extent_is_inline_constant():
    dom, param_map = to_domain((8, 4))
    assert dom.dim(isl.dim_type.PARAM) == 0
    assert dom.dim(isl.dim_type.SET) == 2
    assert param_map == {}


def test_to_domain_bare_dimvar_uses_own_name():
    dom, param_map = to_domain((P,))
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert dom.get_dim_name(isl.dim_type.PARAM, 0) == "P"
    assert param_map == {"P": P}


def test_to_domain_composite_mints_opaque_param_with_dim_range_bound():
    d = simplify_dim(DimFloorDiv, (P, 4))
    dom, param_map = to_domain((d,))
    assert dom.dim(isl.dim_type.PARAM) == 1
    name = dom.get_dim_name(isl.dim_type.PARAM, 0)
    lo, hi = dim_range(d)
    assert f"{lo} <= {name} <= {hi - 1}" in str(dom)
    assert param_map[name] is d


def test_to_domain_same_canonical_expr_dedups_across_axes():
    d = simplify_dim(DimFloorDiv, (P, 4))
    dom, param_map = to_domain((d, 128, d))
    assert dom.dim(isl.dim_type.PARAM) == 1
    assert len(param_map) == 1


def test_to_domain_same_name_conflicting_bounds_raises():
    with pytest.raises(ValueError, match="conflicting bounds"):
        to_domain((DimVar("S", 1, 8), DimVar("S", 1, 16)))


def test_to_domain_rank0():
    dom, param_map = to_domain(())
    assert dom.dim(isl.dim_type.SET) == 0
    assert param_map == {}


# ─── to_dim ─────────────────────────────────────────────────────────────────


def test_to_dim_int():
    assert to_dim(isl.pw_aff("{ [42] }"), {}) == 42


def test_to_dim_id():
    pa = isl.pw_aff("[P] -> { [P] }")
    assert to_dim(pa, {"P": P}) is P


def test_to_dim_unknown_id_raises():
    pa = isl.pw_aff("[P] -> { [P] }")
    with pytest.raises(ValueError, match="no known ShapeDim"):
        to_dim(pa, {})


@pytest.mark.parametrize(
    ("expr_str", "expected_op", "operand"),
    [
        ("(127 + P)", DimAdd, 127),
        ("(P - 3)", DimSub, 3),
        ("(3 * P)", DimMul, 3),
    ],
)
def test_to_dim_binary_ops(expr_str, expected_op, operand):
    pa = isl.pw_aff(f"[P] -> {{ [{expr_str}] }}")
    result = to_dim(pa, {"P": P})
    assert result == simplify_dim(expected_op, (P, operand)) or result == simplify_dim(
        expected_op, (operand, P)
    )


def test_to_dim_minus():
    pa = isl.pw_aff("[P] -> { [(-P)] }")
    assert to_dim(pa, {"P": P}) == simplify_dim(DimSub, (0, P))


def test_to_dim_floordiv():
    # `mod` decomposes into sub/mul/fdiv_q (each already covered above), so
    # it has no separate bare ast node to test this way.
    pa = isl.pw_aff("[P] -> { [floor(P/4)] }")
    assert to_dim(pa, {"P": P}) == simplify_dim(DimFloorDiv, (P, 4))


def test_to_dim_unsupported_op_raises():
    isl.options_set_ast_build_detect_min_max(0)
    try:
        pa = isl.pw_aff("[P, Q] -> { [P] : P > Q; [Q] : P <= Q }")
        with pytest.raises(NotImplementedError):
            to_dim(pa, {"P": P, "Q": Q})
    finally:
        isl.options_set_ast_build_detect_min_max(1)


# ─── round trip ─────────────────────────────────────────────────────────────


def test_round_trip_lossless_for_every_dim_kind():
    dims = (
        128,
        P,
        simplify_dim(DimAdd, (128, P)),
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
