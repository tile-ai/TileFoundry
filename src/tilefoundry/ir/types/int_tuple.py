"""IntTuple alias and the CuTe ``int_tuple.hpp`` / ``tuple_algorithms.hpp`` functions.

A plain ``int`` is a CuTe static value; a dimension expression is a dynamic one.
Functions whose result depends on mode order take ``major``: ``"row"`` reverses
every level of the inputs, calls the function with ``"col"`` (the CuTe body) and
reverses the result back. See [shard §1](docs/spec/shard.md#1-inttuple).
"""

from __future__ import annotations

from typing import Union

from tilefoundry import services
from tilefoundry.ir import expr

from .topology import Topology

__all__ = [
    "IntTuple",
    "append",
    "ceil_div",
    "colex_geq",
    "colex_gtr",
    "colex_leq",
    "colex_less",
    "compatible",
    "compatible_profile",
    "dim_at_most",
    "congruent",
    "depth",
    "elem_geq",
    "elem_gtr",
    "elem_leq",
    "elem_less",
    "elem_scale",
    "evenly_divides",
    "exchange_sort",
    "filter_zeros",
    "flatten",
    "flatten_to_tuple",
    "get",
    "group",
    "inner_product",
    "lex_geq",
    "lex_gtr",
    "lex_leq",
    "lex_less",
    "mode_axes",
    "prepend",
    "product",
    "product_each",
    "product_like",
    "rank",
    "repeat_like",
    "reverse_modes",
    "round_up",
    "safe_div",
    "select",
    "shape_div",
    "size",
    "sum",
    "take",
    "transform_leaf",
    "unflatten",
    "unwrap",
    "weakly_congruent",
    "wrap",
    "zip2_by",
]

IntTuple = Union[int, tuple["IntTuple", ...]]


def reverse_modes(t):
    """The modes of *t* in reverse order at every level; a scalar is returned as it is.

    The project's row-major order: a ``Layout`` leaf reverses its shape and strides.
    """
    if isinstance(t, tuple):
        return tuple(reverse_modes(item) for item in reversed(t))
    if hasattr(t, "strides") and hasattr(t, "shape"):
        return type(t)(reverse_modes(t.shape), reverse_modes(t.strides))
    return t


def rank(t, *indices):
    """CuTe ``int_tuple.hpp`` ``rank``: the number of top-level modes; a scalar has one."""
    if indices:
        return rank(get(t, *indices))
    return len(t) if isinstance(t, tuple) else 1


def depth(t, *indices):
    """CuTe ``int_tuple.hpp`` ``depth``: how deeply *t* nests; a scalar has depth 0."""
    if indices:
        return depth(get(t, *indices))
    if not isinstance(t, tuple):
        return 0
    return 1 + max((depth(item) for item in t), default=0)


def get(t, *indices):
    """CuTe ``int_tuple.hpp`` ``get<I0,I1,...>``: the mode at each index in turn."""
    for index in indices:
        if isinstance(t, tuple):
            t = t[index]
        elif index != 0:
            raise IndexError(f"get: index {index} of a scalar")
    return t


def take(t, b, e):
    """CuTe ``tuple_algorithms.hpp`` ``take<B,E>``: the modes in ``[b, e)``; ``e=-1`` is the end."""
    if e == -1:
        e = rank(t)
    return tuple(get(t, index) for index in range(b, e))


def select(t, *indices):
    """CuTe ``tuple_algorithms.hpp`` ``select<I...>``: the modes at *indices*."""
    return tuple(get(t, index) for index in indices)


def wrap(t):
    """CuTe ``tuple_algorithms.hpp`` ``wrap``: *t* as a tuple."""
    return t if isinstance(t, tuple) else (t,)


def unwrap(t):
    """CuTe ``tuple_algorithms.hpp`` ``unwrap``: remove every one-mode tuple around *t*."""
    if isinstance(t, tuple) and len(t) == 1:
        return unwrap(t[0])
    return t


def flatten_to_tuple(t) -> tuple:
    """CuTe ``tuple_algorithms.hpp`` ``flatten_to_tuple``: every leaf of *t*, as a tuple."""
    if not isinstance(t, tuple):
        return (t,)
    return tuple(value for item in t for value in flatten_to_tuple(item))


def flatten(t):
    """CuTe ``tuple_algorithms.hpp`` ``flatten``: every leaf of a tuple; a scalar stays a scalar."""
    return flatten_to_tuple(t) if isinstance(t, tuple) else t


def take_nested(flat: tuple, profile) -> tuple:
    """Take *profile*'s worth of *flat*, returning it nested and what is left."""
    if not isinstance(profile, tuple):
        if not flat:
            raise ValueError("unflatten: the profile asks for more modes than the tuple has")
        return flat[0], flat[1:]
    nested: list = []
    for item in profile:
        value, flat = take_nested(flat, item)
        nested.append(value)
    return tuple(nested), flat


def unflatten(flat: tuple, profile) -> tuple:
    """CuTe ``tuple_algorithms.hpp`` ``unflatten``: nest a flat tuple to *profile*'s structure.

    Only *profile*'s nesting is read, never its leaves, so the profile may be
    the grouping itself. ``flatten(unflatten(t, p)) == t``.
    """
    flat = wrap(flat)
    nested, rest = take_nested(flat, profile)
    if rest:
        raise ValueError(
            f"unflatten: the profile accounts for {len(flat) - len(rest)} of the "
            f"tuple's {len(flat)} modes"
        )
    return nested


def repeat_like(profile, value) -> object:
    """CuTe ``tuple_algorithms.hpp`` ``repeat_like``: *value* at each leaf of *profile*."""
    if not isinstance(profile, tuple):
        return value
    return tuple(repeat_like(item, value) for item in profile)


def group(t, b, e):
    """CuTe ``tuple_algorithms.hpp`` ``group<B,E>``: the modes in ``[b, e)`` as one mode."""
    t = wrap(t)
    if e == -1:
        e = len(t)
    return (*t[:b], *((take(t, b, e),) if b < e else ()), *t[e:])


def append(t, x, n=None):
    """CuTe ``tuple_algorithms.hpp`` ``append``: *x* after *t*; with *n*, pad *t* to *n* modes."""
    if n is None:
        return (*t, x) if isinstance(t, tuple) else (t, x)
    t = wrap(t) if n > 1 or isinstance(t, tuple) else t
    if not isinstance(t, tuple) or n == len(t):
        return t
    return (*t, *((x,) * (n - len(t))))


def prepend(t, x, n=None):
    """CuTe ``tuple_algorithms.hpp`` ``prepend``: *x* before *t*; with *n*, pad *t* to *n* modes."""
    if n is None:
        return (x, *t) if isinstance(t, tuple) else (x, t)
    t = wrap(t) if n > 1 or isinstance(t, tuple) else t
    if not isinstance(t, tuple) or n == len(t):
        return t
    return (*((x,) * (n - len(t))), *t)


def product(t) -> "ShapeDim":
    """CuTe ``int_tuple.hpp`` ``product``: every leaf multiplied, folded from one.

    A ``Topology`` leaf counts as its size.
    """
    result = 1
    for v in flatten_to_tuple(t):
        if isinstance(v, Topology):
            v = v.size
        result *= v
    return result


def product_each(t) -> tuple:
    """CuTe ``int_tuple.hpp`` ``product_each``: the product of each top-level mode."""
    return tuple(product(mode) for mode in wrap(t))


def product_like(t, guide):
    """CuTe ``int_tuple.hpp`` ``product_like``: the product of *t* at each leaf of *guide*."""
    if isinstance(guide, tuple):
        return tuple(product_like(mode, one) for mode, one in zip(t, guide, strict=True))
    return product(t)


def size(t, *indices):
    """CuTe ``int_tuple.hpp`` ``size``: the product of the mode at *indices*."""
    return product(get(t, *indices))


def sum(t):
    """CuTe ``int_tuple.hpp`` ``sum``: every leaf added."""
    if not isinstance(t, tuple):
        return t
    result = 0
    for item in t:
        result += sum(item)
    return result


def inner_product(a, b):
    """CuTe ``int_tuple.hpp`` ``inner_product``: the sum of leafwise products."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            raise ValueError(f"inner_product: mismatched ranks {len(a)} and {len(b)}")
        result = 0
        for x, y in zip(a, b):
            result += inner_product(x, y)
        return result
    return a * b


def ceil_div(a, b, *, major="row"):
    """CuTe ``int_tuple.hpp`` ``ceil_div``; a tuple divided by an int folds from mode 0.

    Missing modes of a tuple *b* count as one.
    """
    if major == "row":
        return reverse_modes(ceil_div(reverse_modes(a), reverse_modes(b), major="col"))
    if isinstance(a, tuple):
        if isinstance(b, tuple):
            if len(a) < len(b):
                raise ValueError(f"ceil_div: mismatched ranks {len(a)} and {len(b)}")
            padded = append(b, 1, len(a))
            return tuple(ceil_div(x, y, major="col") for x, y in zip(a, padded))
        result = []
        for item in a:
            result.append(ceil_div(item, b, major="col"))
            b = ceil_div(b, item, major="col")
        return tuple(result)
    if isinstance(b, tuple):
        return ceil_div(a, product(b), major="col")
    x, y = expr.static_dim_value(a), expr.static_dim_value(b)
    if x is not None and y is not None:
        return (x + y - 1) // y
    if y == 1:
        return a
    quotient = expr.exact_quotient(a, b)
    if quotient is not None and dim_at_most(1, b):
        return quotient
    if dim_at_most(1, a) and dim_at_most(a, b):
        return 1
    return (a + b - 1) // b


def dim_at_most(a, b) -> bool:
    """Whether ``a <= b`` is proven: by static values, else by the range of ``b - a``.

    A difference with no affine range proves nothing.
    """
    x, y = expr.static_dim_value(a), expr.static_dim_value(b)
    if x is not None and y is not None:
        return x <= y
    try:
        bounds = services.dim_range(services.simplify_dim("sub", (b, a)))
    except (NotImplementedError, TypeError, ValueError):
        return False
    return bounds is not None and bounds[0] >= 0


def round_up(a, b):
    """CuTe ``int_tuple.hpp`` ``round_up``: *a* rounded up to a multiple of *b*, leafwise."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) < len(b):
            raise ValueError(f"round_up: mismatched ranks {len(a)} and {len(b)}")
        return tuple(round_up(x, y) for x, y in zip(a, append(b, 1, len(a))))
    return ((a + b - 1) // b) * b


def shape_div(a, b, *, major="row"):
    """CuTe ``int_tuple.hpp`` ``shape_div``; a tuple divided by an int folds from mode 0.

    Static extents require the CuTe divisibility condition. Symbolic extents are a
    project extension (CuTe divides only integers): equal extents divide to one, a
    product divides by either factor to the other, and any other pair is refused.
    """
    if major == "row":
        return reverse_modes(shape_div(reverse_modes(a), reverse_modes(b), major="col"))
    if isinstance(a, tuple):
        if isinstance(b, tuple):
            return tuple(shape_div(x, y, major="col") for x, y in zip(a, b, strict=True))
        result = []
        for extent in a:
            result.append(shape_div(extent, b, major="col"))
            b = shape_div(b, extent, major="col")
        return tuple(result)
    if isinstance(b, tuple):
        return shape_div(a, product(b), major="col")
    x, y = expr.static_dim_value(a), expr.static_dim_value(b)
    if x is not None and y is not None:
        if x % y and y % x:
            raise ValueError(f"shape_div: {x} and {y} do not divide one another")
        return (x + y - 1) // y
    if compatible(a, b):
        return 1
    quotient = expr.exact_quotient(a, b)
    if quotient is None:
        raise ValueError(f"shape_div: cannot divide symbolic extents {a!r} and {b!r}")
    return quotient


def safe_div(t, u):
    """CuTe ``numeric/math.hpp`` ``safe_div``: ``t / u``, which the caller knows divides.

    A symbolic product with *u* as a factor gives its other factor, as ``shape_div``
    does, once *u* is proven positive; else the quotient stays ``t // u``.
    """
    if isinstance(t, int) and isinstance(u, int):
        return t // u
    quotient = expr.exact_quotient(t, u)
    return quotient if quotient is not None and dim_at_most(1, u) else t // u


def elem_scale(a, b):
    """CuTe ``int_tuple.hpp`` ``elem_scale``: each leaf of *a* times the size of *b* there."""
    if isinstance(a, tuple):
        return tuple(elem_scale(x, y) for x, y in zip(a, b, strict=True))
    return a * product(b)


def congruent(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``congruent``: the same nesting, ignoring the leaves."""
    if isinstance(a, tuple) != isinstance(b, tuple):
        return False
    if not isinstance(a, tuple):
        return True
    return len(a) == len(b) and all(congruent(x, y) for x, y in zip(a, b))


def weakly_congruent(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``weakly_congruent``: *a*'s nesting is a coarsening of *b*'s."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(weakly_congruent(x, y) for x, y in zip(a, b))
    return not isinstance(a, tuple)


def compatible(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``compatible``: each leaf of *a* equals the size of *b* there."""
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(
            compatible(x, y) for x, y in zip(a, b, strict=True)
        )
    b = product(b) if isinstance(b, tuple) else b
    if a is b or a == b:
        return True
    try:
        return services.normalize_dim(services.simplify_dim("sub", (a, b))) == 0
    except (TypeError, ValueError, NotImplementedError):
        return False


def evenly_divides(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``evenly_divides``: tiler *b* divides shape *a* without a rest."""
    if isinstance(b, tuple):
        if rank(b) > rank(a):
            return False
        return all(evenly_divides(get(a, i), one) for i, one in enumerate(b))
    return size(a) == size(b) * size(ceil_div(a, b, major="col"))


def filter_zeros(a, b=None):
    """CuTe ``int_tuple.hpp`` ``filter_zeros``: a one in *b* wherever *a* holds a static 0."""
    if b is None:
        b = a
    if isinstance(a, tuple):
        return tuple(filter_zeros(x, y) for x, y in zip(a, b, strict=True))
    if expr.static_dim_value(a) == 0:
        return repeat_like(b, 1)
    return b


def exchange_sort(keys: list, values: list) -> list:
    """CuTe ``int_tuple.hpp`` ``detail::exchange_sort`` of key/value pairs, by key: the values."""
    pairs = list(zip(keys, values, strict=True))
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if pairs[j][0] < pairs[i][0]:
                pairs[i], pairs[j] = pairs[j], pairs[i]
    return [value for _, value in pairs]


def lex_less(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``lex_less``: lexicographic order, mode 0 first."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        for x, y in zip(a, b):
            if lex_less(x, y):
                return True
            if x != y:
                return False
        return len(a) < len(b)
    return a < b


def lex_leq(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``lex_leq``."""
    return not lex_less(b, a)


def lex_gtr(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``lex_gtr``."""
    return lex_less(b, a)


def lex_geq(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``lex_geq``."""
    return not lex_less(a, b)


def colex_less(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``colex_less``: colexicographic order, the last mode first."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        for x, y in zip(reversed(a), reversed(b)):
            if colex_less(x, y):
                return True
            if x != y:
                return False
        return len(a) < len(b)
    return a < b


def colex_leq(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``colex_leq``."""
    return not colex_less(b, a)


def colex_gtr(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``colex_gtr``."""
    return colex_less(b, a)


def colex_geq(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``colex_geq``."""
    return not colex_less(a, b)


def elem_less(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``elem_less``: every leaf of *a* below the one of *b*."""
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) > len(b):
            return False
        return all(elem_less(x, y) for x, y in zip(a, b))
    return a < b


def elem_leq(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``elem_leq``."""
    return not elem_less(b, a)


def elem_gtr(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``elem_gtr``."""
    return elem_less(b, a)


def elem_geq(a, b) -> bool:
    """CuTe ``int_tuple.hpp`` ``elem_geq``."""
    return not elem_less(a, b)


def transform_leaf(t, f):
    """CuTe ``tuple_algorithms.hpp`` ``transform_leaf``: *f* of each leaf, nested like *t*."""
    if isinstance(t, tuple):
        return tuple(transform_leaf(item, f) for item in t)
    return f(t)


def zip2_by(t, guide):
    """CuTe ``tuple_algorithms.hpp`` ``zip2_by``: gather the rank-2 modes at *guide*'s leaves."""
    if not isinstance(guide, tuple):
        if rank(t) != 2:
            raise ValueError(f"zip2_by: a terminal mode needs rank 2, got {t!r}")
        return t
    if len(t) < len(guide):
        raise ValueError(f"zip2_by: mismatched ranks {len(t)} and {len(guide)}")
    split = tuple(zip2_by(t[i], one) for i, one in enumerate(guide))
    return (
        tuple(get(part, 0) for part in split),
        (*(get(part, 1) for part in split), *t[len(guide):]),
    )


def compatible_profile(authored, shape):
    """Nest authored modes into logical axes without splitting or coalescing."""
    if any(isinstance(mode, tuple) for mode in authored) and compatible(shape, authored):
        return authored
    extents = flatten_to_tuple(authored)
    groups, cursor = [], 0
    try:
        for extent in shape:
            group, accumulated = [], 1
            if expr.static_dim_value(extent) == 1 and cursor < len(extents) and expr.static_dim_value(extents[cursor]) == 1:
                group.append(extents[cursor])
                cursor += 1
            while not compatible(extent, accumulated):
                mode = extents[cursor]
                left, right = expr.static_dim_value(accumulated), expr.static_dim_value(mode)
                accumulated = left * right if left is not None and right is not None else (
                    mode if accumulated == 1 else services.simplify_dim("mul", (accumulated, mode))
                )
                a, b = expr.static_dim_value(accumulated), expr.static_dim_value(extent)
                if a is not None and b is not None and a > b:
                    raise ValueError("mode crosses a logical axis")
                group.append(mode)
                cursor += 1
            groups.append(tuple(group))
        if any(expr.static_dim_value(mode) != 1 for mode in extents[cursor:]):
            raise ValueError("unassigned layout modes")
        if groups:
            groups[-1] += extents[cursor:]
    except (IndexError, ValueError) as error:
        raise ValueError(f"layout modes {extents} cross logical axes of {tuple(shape)}") from error
    return tuple(group[0] if len(group) == 1 else group for group in groups)


def mode_axes(profile) -> tuple[int, ...]:
    """Each flat mode's logical axis, in authored order."""
    return flatten_to_tuple(tuple(repeat_like(mode, axis) for axis, mode in enumerate(profile)))
