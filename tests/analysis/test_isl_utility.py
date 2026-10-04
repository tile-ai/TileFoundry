"""isl_interop — range queries and conversion between IR dimensions and isl."""

from __future__ import annotations

import isl
import pytest

from tests.fixtures.placed.persistent_gemm_tiled import BX, PersistentGemmTiled
from tilefoundry.ir.core import RangeMetadata, attach_metadata
from tilefoundry.ir.core.expr import Call, Var
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord
from tilefoundry.ir.isl_interop import (
    dim_range,
    isl_to_dim,
    normalize_dim,
    shape_to_isl_set,
)
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
from tilefoundry.utils.isl_utils import cardinality

P = DimVar("P", 2048, 1_048_576)
Q = DimVar("Q", 2, 32)


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
        (simplify_dim(DimMul, (P, Q)), DimVar("G", 1, 64)),
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
    assert dim_range(P) == (P.lo, P.hi + 1)
    assert dim_range(simplify_dim(DimAdd, (128, P))) == (128 + P.lo, 128 + P.hi + 1)
    assert dim_range(simplify_dim(DimSub, (P, Q))) == (P.lo - Q.hi, P.hi - Q.lo + 1)
    assert dim_range(simplify_dim(DimMul, (4, P))) == (4 * P.lo, 4 * P.hi + 1)
    assert dim_range(simplify_dim(DimMul, (P, Q))) == (
        P.lo * Q.lo,
        P.hi * Q.hi + 1,
    )
    assert dim_range(simplify_dim(DimFloorDiv, (P, 4))) == (P.lo // 4, P.hi // 4 + 1)
    assert dim_range(simplify_dim(DimMod, (P, 128))) == (0, 128)
    assert dim_range(simplify_dim(DimMax, (P, Q))) == (max(P.lo, Q.lo), max(P.hi, Q.hi) + 1)
    assert dim_range(simplify_dim(DimMin, (P, Q))) == (min(P.lo, Q.lo), min(P.hi, Q.hi) + 1)

    inner = simplify_dim(DimFloorDiv, (P, 4))
    outer = simplify_dim(DimFloorDiv, (inner, 2))
    ilo, ihi = dim_range(inner)
    assert dim_range(outer) == (ilo // 2, (ihi - 1) // 2 + 1)

    product = simplify_dim(DimMul, (P, Q))
    shared_leaf = simplify_dim(DimAdd, (product, P))
    plo, phi = dim_range(product)
    assert dim_range(shared_leaf) == (plo + P.lo, phi + P.hi)


def test_dim_range_symbolic_divisor_unsupported():
    n = DimVar("N", 1, 7)
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        dim_range(simplify_dim(DimFloorDiv, (P, n)))
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        dim_range(simplify_dim(DimMod, (P, n)))


def test_dim_range_prefers_metadata_and_unknown_leaves_return_none():
    unknown = Var(type=TensorType.umat_scalar(), name="runtime")
    assert dim_range(unknown) is None

    attach_metadata(unknown, RangeMetadata(3, 11))
    assert dim_range(unknown) == (3, 11)

    outer = PersistentGemmTiled.entry_function().body.body
    assert isinstance(outer, LoopRegion)

    def calls(value):
        if not isinstance(value, Call):
            return ()
        return (value, *(nested for arg in value.args for nested in calls(arg)))

    coordinate = next(call for call in calls(outer.start) if isinstance(call.target, MeshCoord))
    assert dim_range(coordinate) == (0, BX)


def test_cardinality_maximizes_small_parameter_boxes_exactly():
    interior_maximum = isl.set("[c] -> { [p] : 0 <= c <= 8 and 0 <= p and p < c and p < 8 - c }")
    too_large = isl.set("[x, y] -> { [p] : 0 <= x < 132 and 0 <= y < 132 and p = x + y }")

    assert cardinality(interior_maximum) == 4
    assert cardinality(too_large) is None


def test_cardinality_distinguishes_empty_and_unbounded_parameter_contexts():
    empty = isl.set("[c] -> { [i] : 0 <= i < 4 and c >= 3 and c <= 1 }")
    unbounded = isl.set("[c] -> { [i] : i = 0 }")

    assert cardinality(empty) == 0
    assert cardinality(unbounded) is None


def test_shape_to_isl_set_encoding():
    """Static extents inline.

    Static extents inline; a bare DimVar is a parameter bounded by its envelope;
    a composite mints one opaque param bounded by ``dim_range``, shared across
    axes holding the same object. Every parameter is named into ``values``.
    """
    values = {}
    dom = shape_to_isl_set((8, 4), values)
    assert dom.dim(isl.dim_type.PARAM) == 0
    assert dom.dim(isl.dim_type.SET) == 2
    assert values == {}

    dom = shape_to_isl_set((P,), values)
    name = dom.get_dim_name(isl.dim_type.PARAM, 0)
    assert values == {name: P}
    assert f"{P.lo} <= {name} <= {P.hi}" in str(dom)

    values = {}
    d = simplify_dim(DimFloorDiv, (P, 4))
    dom = shape_to_isl_set((d, 128, d), values)
    assert dom.dim(isl.dim_type.PARAM) == 1
    name = dom.get_dim_name(isl.dim_type.PARAM, 0)
    lo, hi = dim_range(d)
    assert f"{lo} <= {name} <= {hi - 1}" in str(dom)
    assert values == {name: d}


def test_shape_to_isl_set_names_each_value_once():
    """One object is one parameter; two DimVars sharing a name are two."""
    narrow, wide = DimVar("S", 1, 7), DimVar("S", 1, 15)
    values = {}
    dom = shape_to_isl_set((narrow, wide), values)
    assert dom.dim(isl.dim_type.PARAM) == 2
    assert sorted(values.values(), key=lambda dim: dim.hi) == [narrow, wide]

    again = shape_to_isl_set((wide,), values)
    assert again.dim(isl.dim_type.PARAM) == 1
    assert values[again.get_dim_name(isl.dim_type.PARAM, 0)] is wide
    assert len(values) == 2


def test_shape_to_isl_set_literal_shapes():
    """A literal shape is a box, a zero-dim shape one point, a negative one nothing."""
    assert cardinality(shape_to_isl_set((8, 4), {})) == 32
    assert cardinality(shape_to_isl_set((), {})) == 1
    assert shape_to_isl_set((0, 3), {}).is_empty()
    assert shape_to_isl_set((-1, 3), {}).is_empty()
    with pytest.raises(TypeError, match="bool"):
        shape_to_isl_set((True,), {})


def test_round_trip_lossless_for_every_dim_kind():
    """Encode every kind of ``ShapeDim`` into a domain and read it back out.

    The decode side is asserted here rather than on its own, because what matters
    is that it is the exact inverse: a constant comes back an ``int``, a parameter
    comes back the very same ``DimVar`` object, and a parameter with no entry in
    the map is refused instead of being invented as an opaque dim nothing can
    resolve later.
    """
    assert isl_to_dim(isl.pw_aff("{ [42] }"), {}) == 42
    named = isl.pw_aff("[P] -> { [P] }")
    assert isl_to_dim(named, {"P": P}) is P
    with pytest.raises(ValueError, match="no known ShapeDim"):
        isl_to_dim(named, {})

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
    values = {}
    domain = shape_to_isl_set(dims, values)
    recovered = tuple(
        isl_to_dim(domain.dim_max(i).add_constant(1), values) for i in range(len(dims))
    )
    assert recovered == dims
