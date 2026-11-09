"""isl_interop — range queries and conversion between IR dimensions and isl."""

from __future__ import annotations

import isl
import pytest

from tilefoundry import services, utils
from tilefoundry.interfaces import access
from tilefoundry.ir import dim, expr, types
from tilefoundry.ir.hir import math as hir_math

P = expr.DimVar("P", 2048, 1_048_576)


Q = expr.DimVar("Q", 2, 32)


def test_normalize_dim_uses_isl_affine_normal_form():
    verbose = services.simplify_dim(
        "floordiv",
        (
            services.simplify_dim(
                "add",
                (services.simplify_dim("sub", (services.simplify_dim("add", (P, 0)), 0)), 0),
            ),
            1,
        ),
    )
    quotient = services.simplify_dim("floordiv", (services.simplify_dim("add", (P, 8)), 4))
    expected = services.simplify_dim(
        "add",
        (services.simplify_dim("floordiv", (P, 4)), 2),
    )

    constant = services.simplify_dim("mul", (services.simplify_dim("add", (4, 2)), 3))

    assert services.normalize_dim(constant) == 18
    assert isinstance(services.normalize_dim(constant), int)
    assert services.normalize_dim(verbose) is P
    assert services.normalize_dim(quotient) == expected


def test_normalize_dim_leaves_unsupported_expressions_unchanged():
    symbolic_divisor = services.simplify_dim(
        "floordiv",
        (services.simplify_dim("mul", (P, Q)), expr.DimVar("G", 1, 64)),
    )
    piecewise = services.simplify_dim("min", (P, 8192))

    assert services.normalize_dim(symbolic_divisor) is symbolic_divisor
    assert services.normalize_dim(piecewise) is piecewise


def test_normalize_dim_keys_runtime_parameters_by_object_identity():
    scalar = types.TensorType.from_umat_scalar()
    first = expr.Var(type=scalar, name="start")
    second = expr.Var(type=scalar, name="start")

    def distance(left, right):
        return services.simplify_dim(
            "sub",
            (
                services.simplify_dim("add", (left, 9)),
                services.simplify_dim("add", (right, 1)),
            ),
        )

    assert services.normalize_dim(distance(first, first)) == 8
    distinct = services.normalize_dim(distance(first, second))
    assert isinstance(distinct, expr.Call)

    def vars_in(value):
        if isinstance(value, expr.Var):
            return [value]
        if isinstance(value, expr.Call):
            return [leaf for arg in value.args for leaf in vars_in(arg)]
        return []

    leaves = vars_in(distinct)
    assert any(leaf is first for leaf in leaves)
    assert any(leaf is second for leaf in leaves)


def test_dim_range_interval_arithmetic():
    """Conservative half-open interval per dim kind, incl. nesting."""
    assert services.dim_range(7) == (7, 8)
    assert services.dim_range(P) == (P.lo, P.hi + 1)
    assert services.dim_range(services.simplify_dim("add", (128, P))) == (128 + P.lo, 128 + P.hi + 1)
    assert services.dim_range(services.simplify_dim("sub", (P, Q))) == (P.lo - Q.hi, P.hi - Q.lo + 1)
    assert services.dim_range(services.simplify_dim("mul", (4, P))) == (4 * P.lo, 4 * P.hi + 1)
    assert services.dim_range(services.simplify_dim("mul", (P, Q))) == (
        P.lo * Q.lo,
        P.hi * Q.hi + 1,
    )
    assert services.dim_range(services.simplify_dim("floordiv", (P, 4))) == (P.lo // 4, P.hi // 4 + 1)
    assert services.dim_range(services.simplify_dim("mod", (P, 128))) == (0, 128)
    assert services.dim_range(services.simplify_dim("max", (P, Q))) == (max(P.lo, Q.lo), max(P.hi, Q.hi) + 1)
    assert services.dim_range(services.simplify_dim("min", (P, Q))) == (min(P.lo, Q.lo), min(P.hi, Q.hi) + 1)

    inner = services.simplify_dim("floordiv", (P, 4))
    outer = services.simplify_dim("floordiv", (inner, 2))
    ilo, ihi = services.dim_range(inner)
    assert services.dim_range(outer) == (ilo // 2, (ihi - 1) // 2 + 1)

    product = services.simplify_dim("mul", (P, Q))
    shared_leaf = services.simplify_dim("add", (product, P))
    plo, phi = services.dim_range(product)
    assert services.dim_range(shared_leaf) == (plo + P.lo, phi + P.hi)


def test_dim_range_symbolic_divisor_unsupported():
    n = expr.DimVar("N", 1, 7)
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        services.dim_range(services.simplify_dim("floordiv", (P, n)))
    with pytest.raises(NotImplementedError, match="symbolic divisor"):
        services.dim_range(services.simplify_dim("mod", (P, n)))


def test_dim_range_reads_a_constant_product_as_a_divisor():
    """HIR keeps ``GN * GROUP_M`` as a product of constants; it divides like its value."""
    scalar = types.TensorType.from_umat_scalar()
    six = expr.Call(
        type=scalar,
        target=hir_math.Binary(kind=expr.BinaryKind.MUL),
        args=(expr.Constant(value=3, type=scalar), expr.Constant(value=2, type=scalar)),
    )
    assert services.dim_range(services.simplify_dim("floordiv", (P, six))) == (P.lo // 6, P.hi // 6 + 1)
    assert services.dim_range(services.simplify_dim("mod", (P, six))) == (0, 6)


def test_cardinality_maximizes_small_parameter_boxes_exactly():
    interior_maximum = isl.set("[c] -> { [p] : 0 <= c <= 8 and 0 <= p and p < c and p < 8 - c }")
    too_large = isl.set("[x, y] -> { [p] : 0 <= x < 132 and 0 <= y < 132 and p = x + y }")

    assert utils.cardinality(interior_maximum) == 4
    assert utils.cardinality(too_large) is None


def test_cardinality_distinguishes_empty_and_unbounded_parameter_contexts():
    empty = isl.set("[c] -> { [i] : 0 <= i < 4 and c >= 3 and c <= 1 }")
    unbounded = isl.set("[c] -> { [i] : i = 0 }")

    assert utils.cardinality(empty) == 0
    assert utils.cardinality(unbounded) is None


def test_shape_to_isl_set_encoding():
    """Static extents inline.

    Static extents inline; a bare DimVar is a parameter bounded by its envelope;
    a composite mints one opaque param bounded by ``dim_range``, shared across
    axes holding the same object. Every parameter is named into ``values``.
    """
    values = {}
    dom = dim.shape_to_isl_set((8, 4), values)
    assert dom.dim(isl.dim_type.PARAM) == 0
    assert dom.dim(isl.dim_type.SET) == 2
    assert values == {}

    dom = dim.shape_to_isl_set((P,), values)
    name = dom.get_dim_name(isl.dim_type.PARAM, 0)
    assert values == {name: P}
    assert f"{P.lo} <= {name} <= {P.hi}" in str(dom)

    values = {}
    d = services.simplify_dim("floordiv", (P, 4))
    dom = dim.shape_to_isl_set((d, 128, d), values)
    assert dom.dim(isl.dim_type.PARAM) == 1
    name = dom.get_dim_name(isl.dim_type.PARAM, 0)
    lo, hi = services.dim_range(d)
    assert f"{lo} <= {name} <= {hi - 1}" in str(dom)
    assert values == {name: d}


def test_shape_to_isl_set_names_each_value_once():
    """One object is one parameter; two DimVars sharing a name are two.

    A name is a fixed prefix and a number, whatever the value is called, and
    skips any name already taken. An Op's relations are lined up by identity
    the same way, into new relations: one stated relation reused by two Ops
    leaves both its map and its values as they were.
    """
    narrow, wide = expr.DimVar("S", 1, 7), expr.DimVar("S", 1, 15)
    values = {}
    dom = dim.shape_to_isl_set((narrow, wide), values)
    assert dom.dim(isl.dim_type.PARAM) == 2
    assert sorted(values.values(), key=lambda dim_value: dim_value.hi) == [narrow, wide]

    again = dim.shape_to_isl_set((wide,), values)
    assert again.dim(isl.dim_type.PARAM) == 1
    assert values[again.get_dim_name(isl.dim_type.PARAM, 0)] is wide
    assert len(values) == 2

    def fixed(name: str) -> bool:
        return name[0] == "p" and name[1:].isdigit()

    unspellable = expr.DimVar("1 seq-len/ä", 1, 9)
    probe: dict = {}
    dim.shape_to_isl_set((unspellable,), probe)
    (named,) = probe
    assert fixed(named), f"{named!r} is read off the value's own name"
    assert dim.shape_to_isl_set((unspellable,), probe).dim_max_val(0).get_num_si() == 8
    following = int(named[1:]) + 1
    taken = {f"p{following}": narrow, f"p{following + 1}": wide}
    dim.shape_to_isl_set((expr.DimVar("S", 1, 3),), taken)
    (fresh,) = set(taken) - {f"p{following}", f"p{following + 1}"}
    assert fixed(fresh) and taken[f"p{following}"] is narrow and taken[f"p{following + 1}"] is wide

    walk = "0 <= d0 < 8"
    n, other_n, s = expr.DimVar("N", 1, 64), expr.DimVar("N", 1, 32), expr.DimVar("S", 1, 4)
    shared = access.Relation(isl.map(f"[N] -> {{ [d0] -> [d0 + N] : {walk} }}"), {"N": n})
    renamed = access.Relation(isl.map(f"[M] -> {{ [d0] -> [d0 - M] : {walk} }}"), {"M": n})
    homonym = access.Relation(isl.map(f"[N] -> {{ [d0] -> [N] : {walk} }}"), {"N": other_n})
    twice = access.Relation(
        isl.map(f"[A, B] -> {{ [d0] -> [d0 + A - B] : {walk} }}"), {"A": n, "B": n}
    )
    on_coordinate = access.Relation(isl.map(f"[d0] -> {{ [i] -> [i + d0] : {walk} }}"), {"d0": s})
    swapped = (
        access.Relation(isl.map(f"[p0, p1] -> {{ [d0] -> [p0, p1] : {walk} }}"), {"p0": n, "p1": s}),
        access.Relation(isl.map(f"[p0, p1] -> {{ [d0] -> [p0, p1] : {walk} }}"), {"p0": s, "p1": n}),
    )
    stated = (shared, renamed, homonym, twice, on_coordinate, *swapped)
    before = [(acc2.relation, dict(acc2.values)) for acc2 in stated]
    for op in (stated, (shared, homonym, shared)):
        held = access.iterating((8,), op)
        names: dict[int, str] = {}
        for acc in held:
            for name, value in acc.values.items():
                assert names.setdefault(id(value), name) == name, "one value, one name"
        assert len(set(names.values())) == len(names), "two values, two names"
        assert all(name == "N" or fixed(name) for name in names.values())
        assert names[id(n)] != names[id(other_n)]
    held = access.iterating((8,), stated)
    one = held[3].values
    assert len(one) == 1 and next(iter(one.values())) is n, "two names for one value merge"
    assert held[3].relation.is_equal(
        isl.map(f"[{next(iter(one))}] -> {{ [d0] -> [d0] : {walk} }}")
    ), "and merging equates them rather than dropping one"
    first, second = (dict(map(reversed, acc2.values.items())) for acc2 in held[5:])
    assert first == second, "a swap captures neither name"
    assert held[5].relation.is_equal(
        isl.map(f"[{first[n]}, {first[s]}] -> {{ [d0] -> [{first[n]}, {first[s]}] : {walk} }}")
    )
    for acc, (relation, values) in zip(stated, before, strict=True):
        assert acc.relation is relation and acc.values == values


def test_shape_to_isl_set_literal_shapes():
    """A literal shape is a box, a zero-dim shape one point, a negative one nothing."""
    assert utils.cardinality(dim.shape_to_isl_set((8, 4), {})) == 32
    assert utils.cardinality(dim.shape_to_isl_set((), {})) == 1
    assert dim.shape_to_isl_set((0, 3), {}).is_empty()
    assert dim.shape_to_isl_set((-1, 3), {}).is_empty()
    with pytest.raises(TypeError, match="bool"):
        dim.shape_to_isl_set((True,), {})


def test_round_trip_lossless_for_every_dim_kind():
    """Encode every kind of ``ShapeDim`` into a domain and read it back out.

    The decode side is asserted here rather than on its own, because what matters
    is that it is the exact inverse: a constant comes back an ``int``, a parameter
    comes back the very same ``DimVar`` object, and a parameter with no entry in
    the map is refused instead of being invented as an opaque dim nothing can
    resolve later.
    """
    assert dim.isl_to_dimension(isl.pw_aff("{ [42] }"), {}) == 42
    named = isl.pw_aff("[P] -> { [P] }")
    assert dim.isl_to_dimension(named, {"P": P}) is P
    with pytest.raises(ValueError, match="no known ShapeDim"):
        dim.isl_to_dimension(named, {})

    dims = (
        128,
        P,
        services.simplify_dim("add", (128, P)),
        services.simplify_dim("sub", (P, 3)),
        services.simplify_dim("floordiv", (P, 4)),
        services.simplify_dim("floordiv", (services.simplify_dim("floordiv", (P, 4)), 2)),
        services.simplify_dim("mul", (P, Q)),
        services.simplify_dim("mod", (P, 128)),
        services.simplify_dim("add", (128, services.simplify_dim("floordiv", (P, 4)))),
    )
    values = {}
    domain = dim.shape_to_isl_set(dims, values)
    recovered = tuple(
        dim.isl_to_dimension(domain.dim_max(i).add_constant(1), values) for i in range(len(dims))
    )
    assert recovered == dims


@pytest.mark.parametrize(
    "source",
    (
        "{ [d] }",
        "{ [d] : d >= 0 }",
        "[n] -> { [i] : i >= 0 and 0 <= n < 3 }",
    ),
)
def test_cardinality_of_an_unbounded_set_is_unknown(source):
    assert utils.cardinality(isl.set(source)) is None


def test_cardinality_of_a_bounded_parameter_box_stays_exact():
    assert utils.cardinality(isl.set("[n] -> { [i] : 0 <= i < n and 0 <= n < 3 }")) == 2


def test_cardinality_unions_selected_parameters_before_maximizing():
    reached = isl.set("[m, n] -> { [i] : 0 <= m < 3 and 0 <= n <= 2 and 2m <= i < 2m+n }")
    assert utils.cardinality(reached) == 2
    assert utils.cardinality(reached, union=("m",)) == 6
    assert utils.cardinality(reached, union=("m", "n")) == 6


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("[m] -> { [i] : 0 <= m < 5000 and i = m }", 5000),
        ("[m] -> { [i] : i = 0 }", 1),
        ("[m] -> { [i] : i = m }", None),
        ("[m] -> { [i] : false }", 0),
    ),
)
def test_cardinality_union_fallback_and_empty_context(source, expected):
    assert utils.cardinality(isl.set(source), union=("m",)) == expected
