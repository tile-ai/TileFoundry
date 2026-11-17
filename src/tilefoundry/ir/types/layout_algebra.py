"""The CuTe layout algebra of ``layout.hpp``, ``layout_composed.hpp`` and ``swizzle_layout.hpp``.

Each function is the CuTe function of the same name; its overloads for
``Layout``, ``ComposedLayout``, ``Swizzle`` and IntTuple arguments are
dispatched inside it. A function whose result depends on mode order takes
``major``. ``"col"`` is CuTe. ``"row"`` reverses the modes of the inputs, applies
``"col"`` and reverses the result. Structural groupings stay in CuTe order, such
as a divide's ``(tile, rest)``.
See [shard §9](docs/spec/shard.md#9-layout-construction-and-algebra).
"""

from __future__ import annotations

import builtins
from fractions import Fraction
from functools import partial
from math import gcd

from tilefoundry import services
from tilefoundry.ir import expr

from . import int_tuple, swizzle_layout
from .int_tuple import exchange_sort, flatten_to_tuple, reverse_modes, unwrap, wrap
from .layout import (
    ComposedLayout,
    Layout,
    LayoutBase,
    NotProjectableError,
    Swizzle,
    canonical_mode,
    cosize,
    get,
    rank,
    size,
    take,
)
from .stride import LayoutLeft, LayoutRight, compact_major, compact_order, crd2idx
from .swizzle_layout import get_swizzle_portion, inverse, supports_composition, supports_inverse

__all__ = [
    "append",
    "blocked_product",
    "ceil_div",
    "coalesce",
    "complement",
    "composition",
    "dice",
    "domain_distribute",
    "downcast",
    "filter",
    "filter_zeros",
    "flat_divide",
    "flat_product",
    "group",
    "is_contiguous",
    "is_inverse_projectable",
    "left_inverse",
    "logical_divide",
    "logical_product",
    "make_composed_layout",
    "make_fragment_like",
    "make_layout",
    "make_layout_like",
    "make_ordered_layout",
    "max_alignment",
    "max_common_layout",
    "max_common_vector",
    "nullspace",
    "prepend",
    "raked_product",
    "recast_layout",
    "replace",
    "right_inverse",
    "slice",
    "slice_and_offset",
    "tile_to_shape",
    "tile_unzip",
    "tiled_divide",
    "tiled_product",
    "transform_layout",
    "upcast",
    "with_shape",
    "zip",
    "zipped_divide",
    "zipped_product",
]


def make_composed_layout(inner, offset, outer) -> "ComposedLayout":
    """CuTe ``layout_composed.hpp`` ``make_composed_layout(layoutA, offset, layoutB)``."""
    return ComposedLayout(inner, offset, outer)


def make_layout(*args) -> "Layout | ComposedLayout":
    """CuTe ``layout.hpp`` ``make_layout``: from a shape, from layouts, or from a ``Swizzle``.

    ``make_layout(shape)`` is ``LayoutRight``; ``make_layout(shape, LayoutLeft)`` and
    ``make_layout(shape, stride)`` state the strides. ``make_layout(*layouts)``
    keeps each layout as one mode. ``make_layout(swizzle)`` is ``swizzle_layout.hpp``.
    """
    if not args:
        return Layout((), ())
    if len(args) == 1 and isinstance(args[0], Swizzle):
        swizzle = args[0]
        outer = Layout(swizzle.base + swizzle.bits + abs(swizzle.shift), 1)
        return outer if swizzle.bits == 0 else ComposedLayout(swizzle, 0, outer)
    if args and all(isinstance(layout, Layout) for layout in args):
        return Layout(
            tuple(layout.shape for layout in args), tuple(layout.strides for layout in args)
        )
    shape = canonical_mode(args[0])
    stride = args[1] if len(args) > 1 else LayoutRight
    if isinstance(stride, str):
        stride = compact_major(shape, stride)
    return Layout(shape, stride)


def make_ordered_layout(shape, order) -> "Layout":
    """CuTe ``layout.hpp`` ``make_ordered_layout``: compact strides ranked by *order*."""
    return Layout(shape, compact_order(shape, order))


def make_layout_like(layout: "Layout") -> "Layout":
    """CuTe ``layout.hpp`` ``make_layout_like``: compact, in the stride order of *layout*.

    A static stride 0 stays 0.
    """
    return Layout(
        layout.shape,
        compact_order(int_tuple.filter_zeros(layout.strides, layout.shape), layout.strides),
    )


def transform_layout(t0, t1_or_f, f=None):
    """CuTe ``layout.hpp`` ``transform_layout``: one layout of ``f`` of each mode.

    With two inputs, ``f`` takes their common modes and the modes past the shorter
    one follow unchanged, those of *t0* first.
    """
    if f is None:
        return make_layout(*(t1_or_f(get(t0, i)) for i in range(rank(t0))))
    t1 = t1_or_f
    r0, r1 = rank(t0), rank(t1)
    common = min(r0, r1)
    return make_layout(
        *(f(get(t0, i), get(t1, i)) for i in range(common)),
        *(get(t0, i) for i in range(common, r0)),
        *(get(t1, i) for i in range(common, r1)),
    )


def normal(value):
    """*value*, with a symbolic dimension in its normal form."""
    return value if isinstance(value, int) else services.normalize_dim(value)


def bw_coalesce(old_shape, old_stride, new_shape, new_stride):
    """CuTe ``layout.hpp`` ``detail::bw_coalesce``: merge flat modes from the back.

    CuTe merges only static modes. Symbolic modes are a project extension: they
    merge when the normal forms of ``shape * stride`` and the next stride agree.
    """
    for s, d in builtins.zip(reversed(old_shape), reversed(old_stride)):
        if expr.static_dim_value(s) == 1:
            continue
        if not isinstance(new_shape, tuple) and expr.static_dim_value(new_shape) == 1:
            new_shape, new_stride = s, d
            continue
        front_shape, front_stride = int_tuple.get(wrap(new_shape), 0), int_tuple.get(wrap(new_stride), 0)
        reach = normal(s * d)
        merge = reach == normal(front_stride)
        if merge:
            new_shape = replace_front(new_shape, normal(s * front_shape))
            new_stride = replace_front(new_stride, d)
        else:
            new_shape, new_stride = int_tuple.prepend(new_shape, s), int_tuple.prepend(new_stride, d)
    if not isinstance(new_shape, tuple) and expr.static_dim_value(new_shape) == 1:
        return Layout(1, 0)
    return Layout(new_shape, new_stride)


def replace_front(t, x):
    """CuTe ``tuple_algorithms.hpp`` ``replace_front``."""
    return (x, *t[1:]) if isinstance(t, tuple) else x


def coalesce_x(layout: Layout) -> Layout:
    """CuTe ``layout.hpp`` ``detail::coalesce_x``: ``coalesce``, kept past the domain too."""
    shapes = tuple(map(normal, flatten_to_tuple(layout.shape)))
    steps = tuple(map(normal, flatten_to_tuple(layout.strides)))
    last = 2 if expr.static_dim_value(shapes[-1]) == 1 else shapes[-1]
    return bw_coalesce(shapes[:-1], steps[:-1], last, steps[-1])


def coalesce(layout, trg_profile=None, *, major="row"):
    """CuTe ``coalesce``: merge the modes that step on from one another.

    A tuple *trg_profile* coalesces each of its modes and keeps the modes past it.
    A shape (an IntTuple) merges neighbouring modes that are both static or both
    symbolic. ``layout.hpp`` and ``layout_composed.hpp``.
    """
    if isinstance(layout, ComposedLayout):
        return make_composed_layout(
            layout.inner, layout.offset, coalesce(layout.outer, trg_profile, major=major)
        )
    if not isinstance(layout, Layout):
        flat = flatten_to_tuple(layout)
        result = flat[0]
        for extent in flat[1:]:
            back = wrap(result)[-1]
            if (expr.static_dim_value(back) is None) == (expr.static_dim_value(extent) is None):
                result = (*wrap(result)[:-1], back * extent) if isinstance(result, tuple) else back * extent
            else:
                result = int_tuple.append(result, extent)
        return result
    if isinstance(trg_profile, tuple):
        if len(trg_profile) > rank(layout):
            raise ValueError(
                f"coalesce: the profile has {len(trg_profile)} modes, but the layout has {rank(layout)}"
            )
        return transform_layout(layout, trg_profile, partial(coalesce, major=major))
    if major == "row":
        return reverse_modes(coalesce(reverse_modes(layout), major="col"))
    shapes = tuple(map(normal, flatten_to_tuple(layout.shape)))
    steps = tuple(map(normal, flatten_to_tuple(layout.strides)))
    if not shapes:
        return Layout(1, 0)
    return bw_coalesce(shapes[:-1], steps[:-1], shapes[-1], steps[-1])


def filter_zeros(layout, trg_profile=None):
    """CuTe ``filter_zeros``: extent 1 wherever the stride (or *trg_profile*) is a static 0.

    ``layout.hpp``; an IntTuple pair is ``int_tuple.hpp``.
    """
    if isinstance(layout, Layout):
        guide = layout.strides if trg_profile is None else trg_profile
        return Layout(int_tuple.filter_zeros(guide, layout.shape), layout.strides)
    return int_tuple.filter_zeros(layout, trg_profile)


def filter(layout, trg_profile=None, *, major="row"):
    """CuTe ``filter``: drop static-0 strides and extent-1 modes, then coalesce.

    ``layout.hpp`` and ``layout_composed.hpp``. An empty result is ``1:0``.
    """
    if isinstance(layout, ComposedLayout):
        return make_composed_layout(
            layout.inner, layout.offset, filter(layout.outer, trg_profile, major=major)
        )
    if isinstance(trg_profile, tuple):
        if len(trg_profile) > rank(layout):
            raise ValueError(
                f"filter: the profile has {len(trg_profile)} modes, but the layout has {rank(layout)}"
            )
        return transform_layout(layout, trg_profile, partial(filter, major=major))
    return coalesce(filter_zeros(layout), major=major)


def is_contiguous(layout: Layout, *, major: str = "row") -> bool:
    """Whether an already filtered layout covers ``[0, size)`` without gaps.

    Static extents use CuTe's ``size == cosize``. Symbolic extents look for one
    stride-1 mode after coalescing in either direction.
    """
    if all(expr.static_dim_value(extent) is not None for extent in flatten_to_tuple(layout.shape)):
        return size(layout) == cosize(layout)
    if rank(flatten_to_tuple(layout.shape)) == 1 and flatten_to_tuple(layout.strides) == (1,):
        return True
    other = coalesce(layout, major="row" if major == "col" else "col")
    return rank(flatten_to_tuple(other.shape)) == 1 and flatten_to_tuple(other.strides) == (1,)


def append(layout, x=None, n=None):
    """CuTe ``append``: *x* (by default ``1:0``) as a last mode; with *n*, pad to *n* modes.

    ``layout.hpp`` and ``layout_composed.hpp``; an IntTuple is ``tuple_algorithms.hpp``.
    """
    if isinstance(layout, ComposedLayout):
        return make_composed_layout(layout.inner, layout.offset, append(layout.outer, x, n))
    if isinstance(layout, Layout):
        x = Layout(1, 0) if x is None else x
        return Layout(
            int_tuple.append(layout.shape, x.shape, n), int_tuple.append(layout.strides, x.strides, n)
        )
    return int_tuple.append(layout, x, n)


def prepend(layout, x=None, n=None):
    """CuTe ``prepend``: *x* (by default ``1:0``) as a first mode; with *n*, pad to *n* modes."""
    if isinstance(layout, Layout):
        x = Layout(1, 0) if x is None else x
        return Layout(
            int_tuple.prepend(layout.shape, x.shape, n), int_tuple.prepend(layout.strides, x.strides, n)
        )
    return int_tuple.prepend(layout, x, n)


def replace(layout, n, x):
    """CuTe ``replace<N>``: mode *n* is *x*; ``layout.hpp`` and ``tuple_algorithms.hpp``."""
    if isinstance(layout, Layout):
        return Layout(replace(layout.shape, n, x.shape), replace(layout.strides, n, x.strides))
    if not isinstance(layout, tuple):
        if n != 0:
            raise IndexError(f"replace: index {n} of a scalar")
        return x
    return (*layout[:n], x, *layout[n + 1:])


def group(layout, b, e):
    """CuTe ``group<B,E>``: modes ``[b, e)`` as one; ``layout.hpp``, ``layout_composed.hpp``."""
    if isinstance(layout, ComposedLayout):
        return make_composed_layout(layout.inner, layout.offset, group(layout.outer, b, e))
    if isinstance(layout, Layout):
        return Layout(int_tuple.group(layout.shape, b, e), int_tuple.group(layout.strides, b, e))
    return int_tuple.group(layout, b, e)


def composition_impl(lhs_shape, lhs_stride, rhs_shape, rhs_stride) -> Layout:
    """CuTe ``layout.hpp`` ``detail::composition_impl`` for integer strides."""
    if isinstance(rhs_shape, tuple):
        return make_layout(
            *(
                composition_impl(lhs_shape, lhs_stride, s, d)
                for s, d in builtins.zip(rhs_shape, rhs_stride, strict=True)
            )
        ) if rhs_shape else Layout((), ())
    if expr.static_dim_value(rhs_stride) == 0:
        return Layout(rhs_shape, rhs_stride)
    if not isinstance(lhs_shape, tuple):
        return Layout(rhs_shape, scaled(rhs_stride, lhs_stride))
    result_shape, result_stride = (), ()
    rest_shape, rest_stride = rhs_shape, rhs_stride
    for curr_shape, curr_stride in builtins.zip(lhs_shape[:-1], lhs_stride[:-1]):
        if expr.static_dim_value(curr_shape) is not None and expr.static_dim_value(rest_stride) is not None:
            if rest_stride % curr_shape and rest_stride >= curr_shape:
                raise ValueError("composition: Stride Divisibility Condition")
        magnitude = abs(rest_stride) if isinstance(rest_stride, int) else rest_stride
        sign = -1 if isinstance(rest_stride, int) and rest_stride < 0 else 1
        next_shape = int_tuple.ceil_div(curr_shape, magnitude)
        next_stride = scaled(int_tuple.ceil_div(magnitude, curr_shape), sign)
        if expr.static_dim_value(next_shape) == 1 or expr.static_dim_value(rest_shape) == 1:
            rest_stride = next_stride
            continue
        new_shape = least(next_shape, rest_shape, curr_shape, magnitude)
        if expr.static_dim_value(new_shape) is not None and expr.static_dim_value(rest_shape) is not None:
            if rest_shape % new_shape:
                raise ValueError("composition: Shape Divisibility Condition")
        result_shape += (new_shape,)
        result_stride += (scaled(rest_stride, curr_stride),)
        rest_shape = int_tuple.safe_div(rest_shape, new_shape)
        rest_stride = next_stride
    if not result_shape:
        return Layout(rest_shape, scaled(rest_stride, lhs_stride[-1]))
    if expr.static_dim_value(rest_shape) == 1:
        return Layout(unwrap(result_shape), unwrap(result_stride))
    return Layout(
        (*result_shape, rest_shape), (*result_stride, scaled(rest_stride, lhs_stride[-1]))
    )


def least(next_shape, rest_shape, curr_shape, magnitude):
    """CuTe ``min(next_shape, rest_shape)`` of ``detail::composition_impl``.

    A symbolic operand takes the one proven no larger: ``rest_shape`` also when
    ``rest_shape * magnitude <= curr_shape``, which bounds the ceiling
    ``next_shape``. An unproven order stays the dynamic ``min``.
    """
    if expr.static_dim_value(next_shape) is not None and expr.static_dim_value(rest_shape) is not None:
        return min(next_shape, rest_shape)
    if int_tuple.dim_at_most(rest_shape, next_shape) or (
        int_tuple.dim_at_most(1, magnitude)
        and int_tuple.dim_at_most(services.simplify_dim("mul", (rest_shape, magnitude)), curr_shape)
    ):
        return rest_shape
    if int_tuple.dim_at_most(next_shape, rest_shape):
        return next_shape
    return services.simplify_dim("min", (next_shape, rest_shape))


def scaled(value, factor):
    """*value* times *factor*; a static factor 1 or 0 gives *value* or 0, as CuTe ``Int`` does."""
    step = expr.static_dim_value(factor)
    if step == 1:
        return value
    if step == 0:
        return 0
    return value * factor


def composition(lhs, *rhs, major="row"):
    """CuTe ``composition(A, B)``: ``A`` after ``B``; ``composition(A, offset, B)`` is all three.

    *B* may be a ``Layout``, an integer tile, a tuple of per-mode tiles (modes of
    ``A`` past it are dropped) or ``None`` (CuTe ``_``). ``layout.hpp``,
    ``layout_composed.hpp`` and ``swizzle_layout.hpp``.
    """
    if len(rhs) == 2:
        offset, outer = rhs
        if isinstance(lhs, Swizzle) and lhs.bits == 0 and offset == 0 and isinstance(outer, Layout):
            return outer
        return make_composed_layout(lhs, offset, outer)
    (rhs,) = rhs
    if supports_composition(lhs, rhs):
        return swizzle_layout.composition(lhs, rhs, major=major)
    if isinstance(lhs, ComposedLayout):
        return composition(lhs.inner, lhs.offset, composition(lhs.outer, rhs, major=major))
    if isinstance(lhs, Layout) and isinstance(rhs, ComposedLayout):
        if rhs.offset != 0:
            raise ValueError("composition: Require offset == 0.")
        inner = lhs if rhs.inner is None else composition(lhs, rhs.inner, major=major)
        return composition(inner, rhs.outer, major=major)
    if not isinstance(lhs, Layout):
        raise NotImplementedError(
            f"composition: no rule for {type(lhs).__name__} ∘ {type(rhs).__name__}"
        )
    if isinstance(rhs, tuple):
        if len(rhs) > rank(lhs):
            raise ValueError(f"composition: the tiler has {len(rhs)} modes, the layout {rank(lhs)}")
        return make_layout(*(composition(get(lhs, i), one, major=major) for i, one in enumerate(rhs)))
    if rhs is None:
        return lhs
    if major == "row":
        return reverse_modes(composition(reverse_modes(lhs), reverse_modes(rhs), major="col"))
    flat = coalesce_x(lhs)
    if isinstance(rhs, Layout):
        return composition_impl(flat.shape, flat.strides, rhs.shape, rhs.strides)
    return composition_impl(flat.shape, flat.strides, rhs, 1)


def complement(layout, cotarget=None, *, major="row"):
    """CuTe ``complement``: the layout of the indices below *cotarget* that *layout* misses.

    *cotarget* defaults to ``cosize(layout)``. ``layout.hpp`` and ``layout_composed.hpp``.

    Raises:
        NotProjectableError: The layout is not injective.
    """
    if isinstance(layout, ComposedLayout):
        return complement(
            layout.outer, cosize(layout) if cotarget is None else cotarget, major=major
        )
    if major == "row":
        return reverse_modes(
            complement(reverse_modes(layout), reverse_modes(cotarget), major="col")
        )
    filtered = filter(layout, major="col")
    if cotarget is None:
        cotarget = cosize(filtered)
    if not isinstance(filtered.strides, tuple) and expr.static_dim_value(filtered.strides) == 0:
        return make_layout(coalesce(cotarget), LayoutLeft)
    shape, stride = list(wrap(filtered.shape)), list(wrap(filtered.strides))
    if len(stride) > 1 and any(expr.static_dim_value(d) is None for d in stride):
        raise ValueError("complement: Dynamic-stride complement only for rank-1 layouts")
    result_shape, result_stride = [], [1]
    for _ in range(len(stride) - 1):
        min_stride = min(stride)
        min_idx = stride.index(min_stride)
        result_shape.append(min_stride // result_stride[-1])
        result_stride.append(min_stride * shape[min_idx])
        del shape[min_idx], stride[min_idx]
    result_shape.append(stride[0] // result_stride[-1])
    if any(expr.static_dim_value(extent) == 0 for extent in result_shape):
        raise NotProjectableError("complement: Non-injective Layout detected in complement.")
    new_stride = stride[0] * shape[0]
    rest_shape = coalesce(int_tuple.ceil_div(cotarget, new_stride, major="col"))
    rest_stride = compact_major(rest_shape, LayoutLeft, new_stride)
    return coalesce(
        Layout((tuple(result_shape), rest_shape), (tuple(result_stride), rest_stride)), major="col"
    )


def right_inverse(layout, *, major="row"):
    """CuTe ``right_inverse``: ``layout(right_inverse(layout)(i)) == i`` on its domain.

    ``None`` (CuTe ``_``) is its own inverse and an integer offset inverts to its
    negation. ``layout.hpp``, ``layout_composed.hpp`` and ``swizzle_layout.hpp``.
    """
    if layout is None:
        return None
    if isinstance(layout, int):
        return -layout
    if supports_inverse(layout):
        return inverse(layout, partial(right_inverse, major=major), major=major)
    if isinstance(layout, ComposedLayout):
        return composition(
            right_inverse(layout.outer, major=major),
            right_inverse(layout.offset),
            right_inverse(layout.inner, major=major),
        )
    if major == "row":
        return reverse_modes(right_inverse(reverse_modes(layout), major="col"))
    clayout = coalesce(layout, major="col")
    lstride, lshape = wrap(clayout.strides), wrap(clayout.shape)
    preprod = [1]
    for extent in lshape:
        preprod.append(extent * preprod[-1])
    kept = [i for i, d in enumerate(lstride) if expr.static_dim_value(d) is not None]
    result_shape, result_stride, current = [1], [0], 1
    for i in exchange_sort([lstride[i] for i in kept], kept):
        if expr.static_dim_value(current) is not None and lstride[i] == current:
            result_shape.append(lshape[i])
            result_stride.append(preprod[i])
            current = lshape[i] * lstride[i]
    return coalesce(Layout(tuple(result_shape), tuple(result_stride)), major="col")


def left_inverse(layout, *, major="row"):
    """CuTe ``left_inverse``: ``left_inverse(layout)(layout(i)) == i`` for an injective layout.

    A composed mesh layout must have an identity ``inner`` and an
    inverse-projectable ``outer``. ``layout.hpp``, ``layout_composed.hpp`` and
    ``swizzle_layout.hpp``.

    Raises:
        NotProjectableError: A composed layout is not admissible, or the strides
            fail the CuTe divisibility condition.
    """
    if layout is None:
        return None
    if isinstance(layout, int):
        return -layout
    if supports_inverse(layout):
        return inverse(layout, partial(left_inverse, major=major), major=major)
    if isinstance(layout, ComposedLayout):
        check_admissible(layout)
        return composition(
            left_inverse(layout.outer, major=major),
            left_inverse(layout.offset),
            left_inverse(layout.inner, major=major),
        )
    if major == "row":
        return reverse_modes(left_inverse(reverse_modes(layout), major="col"))
    clayout = coalesce(layout, major="col")
    lstride, lshape = wrap(clayout.strides), wrap(clayout.shape)
    if any(expr.static_dim_value(d) is None for d in lstride):
        raise NotProjectableError("left_inverse: Left inverse requires static strides.")
    preprod = [1]
    for extent in lshape:
        preprod.append(extent * preprod[-1])
    order = exchange_sort(list(lstride), list(range(len(lstride))))
    result_shape, result_stride = [], [0]
    for i in order:
        if lstride[i] == 0:
            continue
        reached = int_tuple.product(tuple(result_shape))
        if lstride[i] % reached:
            raise NotProjectableError("left_inverse: Left inverse divisibility condition")
        result_shape.append(lstride[i] // reached)
        result_stride.append(preprod[i])
    return coalesce(
        Layout((*result_shape, lshape[order[-1]]), tuple(result_stride)), major="col"
    )


def is_inverse_projectable(layout: Layout) -> bool:
    """Whether ``layout`` can serve as a mesh execution scope: injective and compact-ordered.

    Drop extent-1 modes and sort the rest by stride. Each stride must be non-zero
    and a multiple of the running codomain extent. ``(5,3):(3,8)`` is injective
    yet not projectable.
    """
    current = 1
    modes = sorted(
        (stride, shape)
        for shape, stride in builtins.zip(
            flatten_to_tuple(layout.shape), flatten_to_tuple(layout.strides)
        )
        if shape != 1
    )
    for stride, shape in modes:
        if stride == 0 or stride % current != 0:
            return False
        current = stride * shape
    return True


def check_admissible(scope: ComposedLayout) -> None:
    """Refuse a composed mesh scope with a non-identity inner or a non-projectable outer."""
    inner = scope.inner
    identity = inner is None or (
        isinstance(inner, Layout)
        and flatten_to_tuple(inner.strides) == compact_major(flatten_to_tuple(inner.shape), LayoutLeft)
    )
    if not identity:
        raise NotProjectableError("non-identity inner is not an admissible mesh scope (v1)")
    if not isinstance(scope.outer, Layout):
        raise NotProjectableError("outer must be a plain Layout for a mesh scope")
    if not is_inverse_projectable(scope.outer):
        raise NotProjectableError("outer layout is not inverse-projectable (injective + compact)")


def zip(layout, *others):
    """CuTe ``zip``: transpose the modes; ``layout.hpp`` and ``tuple_algorithms.hpp``."""
    if isinstance(layout, ComposedLayout):
        return make_composed_layout(layout.inner, layout.offset, zip(layout.outer))
    if isinstance(layout, Layout):
        return Layout(
            zip(layout.shape, *(one.shape for one in others)),
            zip(layout.strides, *(one.strides for one in others)),
        )
    if others:
        return zip((layout, *others))
    if not isinstance(layout, tuple):
        return layout
    if not isinstance(layout[0], tuple):
        return (layout,)
    if any(len(one) != len(layout[0]) for one in layout):
        raise ValueError("zip: mismatched ranks")
    return tuple(tuple(one[j] for one in layout) for j in range(len(layout[0])))


def tile_unzip(layout, tiler):
    """CuTe ``tile_unzip``: zip the rank-2 modes at *tiler*'s leaves into their own modes."""
    if isinstance(layout, ComposedLayout):
        return make_composed_layout(layout.inner, layout.offset, tile_unzip(layout.outer, tiler))
    return Layout(int_tuple.zip2_by(layout.shape, tiler), int_tuple.zip2_by(layout.strides, tiler))


def logical_divide(layout, tiler, *, major="row"):
    """CuTe ``logical_divide``: each tiled mode becomes ``(tile, rest)``.

    *tiler* is a ``Layout``, an integer, a tuple of per-mode tilers (modes past it
    are kept) or ``None``. ``layout.hpp`` and ``layout_composed.hpp``.
    """
    if isinstance(layout, ComposedLayout):
        return make_composed_layout(
            layout.inner, layout.offset, logical_divide(layout.outer, tiler, major=major)
        )
    if isinstance(tiler, tuple):
        if len(tiler) > rank(layout):
            raise ValueError("logical_divide: Too many modes in tiler.")
        return transform_layout(layout, tiler, partial(logical_divide, major=major))
    if tiler is None:
        return layout
    if not isinstance(tiler, Layout):
        tiler = make_layout(tiler, LayoutLeft)
    cotarget = coalesce(layout, major=major).shape
    return composition(
        layout, make_layout(tiler, complement(tiler, cotarget, major=major)), major=major
    )


def ceil_div(target, tiler, *, major="row"):
    """CuTe ``ceil_div``: with a ``Layout`` tiler, the rest shape of ``logical_divide``.

    Otherwise it is the IntTuple ``ceil_div`` of ``int_tuple.hpp``.
    """
    if isinstance(tiler, Layout):
        target = target.shape if isinstance(target, LayoutBase) else target
        return complement(tiler, target, major=major).shape
    return int_tuple.ceil_div(target, tiler, major=major)


def zipped_divide(layout, tiler, *, major="row"):
    """CuTe ``zipped_divide``: ``((tile modes), (rest modes))``."""
    return tile_unzip(logical_divide(layout, tiler, major=major), tiler)


def tiled_divide(layout, tiler, *, major="row"):
    """CuTe ``tiled_divide``: ``((tile modes), rest modes...)``."""
    result = zipped_divide(layout, tiler, major=major)
    rest = rank(result, 1)
    return slice((None, (None,) * rest if rest != 1 else None), result)


def flat_divide(layout, tiler, *, major="row"):
    """CuTe ``flat_divide``: ``(tile modes..., rest modes...)``."""
    result = zipped_divide(layout, tiler, major=major)
    tile, rest = rank(result, 0), rank(result, 1)
    return slice(
        ((None,) * tile if tile != 1 else None, (None,) * rest if rest != 1 else None), result
    )


def logical_product(block, tiler, *, major="row"):
    """CuTe ``logical_product``: ``(block, the block repeated over tiler)``."""
    if isinstance(block, ComposedLayout):
        return make_composed_layout(
            block.inner, block.offset, logical_product(block.outer, tiler, major=major)
        )
    if isinstance(tiler, tuple):
        if len(tiler) > rank(block):
            raise ValueError("logical_product: Too many modes in tiler.")
        return transform_layout(block, tiler, partial(logical_product, major=major))
    if tiler is None:
        return block
    if not isinstance(tiler, Layout):
        tiler = make_layout(tiler, LayoutLeft)
    rest = complement(block, size(block) * cosize(tiler), major=major)
    return make_layout(block, composition(rest, tiler, major=major))


def zipped_product(block, tiler, *, major="row"):
    """CuTe ``zipped_product``: ``((block modes), (product modes))``."""
    return tile_unzip(logical_product(block, tiler, major=major), tiler)


def tiled_product(block, tiler, *, major="row"):
    """CuTe ``tiled_product``: ``((block modes), product modes...)``."""
    result = zipped_product(block, tiler, major=major)
    rest = rank(result, 1)
    return slice((None, (None,) * rest if rest != 1 else None), result)


def flat_product(block, tiler, *, major="row"):
    """CuTe ``flat_product``: ``(block modes..., product modes...)``."""
    result = zipped_product(block, tiler, major=major)
    tile, rest = rank(result, 0), rank(result, 1)
    return slice(
        ((None,) * tile if tile != 1 else None, (None,) * rest if rest != 1 else None), result
    )


def blocked_product(block, tiler, *, major="row"):
    """CuTe ``blocked_product``: each mode is ``(block mode, tiler mode)``."""
    if isinstance(block, ComposedLayout):
        return make_composed_layout(
            block.inner, block.offset, blocked_product(block.outer, tiler, major=major)
        )
    modes = max(rank(block), rank(tiler))
    result = logical_product(append(block, n=modes), append(tiler, n=modes), major=major)
    return zip(get(result, 0), get(result, 1))


def raked_product(block, tiler, *, major="row"):
    """CuTe ``raked_product``: each mode is ``(tiler mode, block mode)``."""
    if isinstance(block, ComposedLayout):
        return make_composed_layout(
            block.inner, block.offset, raked_product(block.outer, tiler, major=major)
        )
    modes = max(rank(block), rank(tiler))
    result = logical_product(append(block, n=modes), append(tiler, n=modes), major=major)
    return zip(get(result, 1), get(result, 0))


def tile_to_shape(block, trg_shape, ord_shape=None, *, major="row"):
    """CuTe ``tile_to_shape``: repeat *block* until it has *trg_shape*.

    *ord_shape* orders the repeats; it defaults to the order of *major*.
    """
    if isinstance(block, ComposedLayout):
        return make_composed_layout(
            block.inner,
            block.offset,
            tile_to_shape(block.outer, trg_shape, ord_shape, major=major),
        )
    if ord_shape is None:
        ord_shape = LayoutLeft if major == "col" else LayoutRight
    if rank(block) > rank(trg_shape):
        raise ValueError("tile_to_shape: Rank of layout must be <= rank of target shape.")
    padded = append(block, n=rank(trg_shape))
    block_shape = int_tuple.product_each(padded.shape)
    target_shape = int_tuple.product_each(trg_shape)
    if all(expr.static_dim_value(one) is not None for one in target_shape):
        if not int_tuple.evenly_divides(target_shape, block_shape):
            raise ValueError("tile_to_shape: block shape does not divide the target shape.")
    product_shape = int_tuple.ceil_div(target_shape, block_shape, major=major)
    return blocked_product(padded, make_ordered_layout(product_shape, ord_shape), major=major)


def make_fragment_like(layout) -> "Layout":
    """CuTe ``make_fragment_like``: a compact layout like *layout*, mode 0 column-major.

    The remaining modes follow the stride order of *layout*. A shape gives
    ``make_layout(shape, LayoutLeft)``. ``layout.hpp`` and ``swizzle_layout.hpp``.
    """
    if get_swizzle_portion(layout) is not None:
        return make_fragment_like(layout.outer)
    if not isinstance(layout, Layout):
        return make_layout(layout, LayoutLeft)
    modes = rank(layout)
    if modes > 1 and all(expr.static_dim_value(one) is not None for one in flatten_to_tuple(layout.shape)):
        shape0, stride0 = get(layout.shape, 0), get(layout.strides, 0)
        return tiled_product(
            Layout(shape0, compact_major(int_tuple.filter_zeros(stride0, shape0), LayoutLeft)),
            make_ordered_layout(take(layout.shape, 1, modes), take(layout.strides, 1, modes)),
            major="col",
        )
    return make_layout(layout.shape, LayoutLeft)


def with_shape(layout, shape, *, major="row"):
    """CuTe ``Layout::with_shape``: *layout* read through a compact layout of *shape*."""
    return composition(layout, make_layout(shape, major), major=major)


def upcast(layout, n: int):
    """CuTe ``upcast<N>``: the layout in units of *n* elements.

    The stride-1 mode shrinks by *n*; every other stride divides by *n*.
    ``layout.hpp``, ``layout_composed.hpp`` and ``swizzle_layout.hpp``.
    """
    if layout is None:
        return None
    if isinstance(layout, Swizzle):
        if n & (n - 1):
            raise ValueError("upcast: N must be a power of two")
        base = layout.base - (n.bit_length() - 1)
        if base >= 0:
            return Swizzle(layout.bits, base, layout.shift)
        return Swizzle(max(layout.bits + base, 0), 0, layout.shift)
    if isinstance(layout, ComposedLayout):
        return composition(upcast(layout.inner, n), upcast(layout.offset, n), upcast(layout.outer, n))
    if not isinstance(layout, Layout):
        return int_tuple.safe_div(layout, n)
    shape, stride = layout.shape, layout.strides
    if isinstance(shape, tuple):
        return make_layout(*(upcast(Layout(s, d), n) for s, d in builtins.zip(shape, stride)))
    if expr.static_dim_value(stride) == 0:
        return layout
    if expr.static_dim_value(stride) is not None:
        if stride % n and n % stride:
            raise ValueError("upcast: Divisibility condition")
        sign = -1 if stride < 0 else 1
        return Layout(
            int_tuple.ceil_div(shape, int_tuple.ceil_div(n, abs(stride))),
            sign * int_tuple.ceil_div(abs(stride), n),
        )
    return Layout(shape, int_tuple.safe_div(stride, n))


def downcast(layout, n: int):
    """CuTe ``downcast<N>``: the layout in units of 1/*n* element.

    The stride-1 mode grows by *n*; every other stride multiplies by *n*.
    """
    if layout is None:
        return None
    if isinstance(layout, Swizzle):
        if n & (n - 1):
            raise ValueError("downcast: N must be a power of two")
        return Swizzle(layout.bits, layout.base + n.bit_length() - 1, layout.shift)
    if isinstance(layout, ComposedLayout):
        return composition(
            downcast(layout.inner, n), downcast(layout.offset, n), downcast(layout.outer, n)
        )
    if not isinstance(layout, Layout):
        return layout * n
    if not any(expr.static_dim_value(d) == 1 for d in flatten_to_tuple(layout.strides)):
        raise ValueError("downcast: Downcast requires adjacent elements")

    def per_mode(shape, stride):
        if isinstance(shape, tuple):
            return make_layout(*(per_mode(s, d) for s, d in builtins.zip(shape, stride)))
        if expr.static_dim_value(stride) in (1, -1):
            return Layout(shape * n, stride)
        return Layout(shape, stride * n)

    return per_mode(layout.shape, layout.strides)


def recast_layout(layout, old_bits: int, new_bits: int):
    """CuTe ``recast_layout<OldType, NewType>``, with the two element widths in bits."""
    scale = Fraction(new_bits, old_bits)
    if scale == 1:
        return layout
    if scale.numerator == 1:
        return downcast(layout, scale.denominator)
    if scale.denominator == 1:
        return upcast(layout, scale.numerator)
    return downcast(upcast(layout, scale.numerator), scale.denominator)


def max_alignment(layout):
    """CuTe ``max_alignment``: the largest *n* that ``upcast`` by *n* accepts.

    A swizzle aligns to ``1 << base``; a composed layout with a swizzle takes the
    gcd of its parts; any other composed layout aligns to 1.
    """
    if isinstance(layout, Swizzle):
        return 1 << layout.base
    if isinstance(layout, int):
        return layout
    if isinstance(layout, ComposedLayout):
        if not isinstance(layout.inner, Swizzle):
            return 1
        return gcd(
            max_alignment(layout.inner), max_alignment(layout.offset), max_alignment(layout.outer)
        )
    flat = coalesce(layout, major="col")
    shape = tuple(s if expr.static_dim_value(s) is not None else 1 for s in wrap(flat.shape))
    stride = tuple(d if expr.static_dim_value(d) is not None else 0 for d in wrap(flat.strides))
    filtered = Layout(unwrap(shape), unwrap(stride))
    permuted = logical_divide(filtered, right_inverse(filtered, major="col"), major="col")
    return gcd(size(permuted, 0), *flatten_to_tuple(get(permuted.strides, 1)))


def max_common_layout(a, b, *, major="row"):
    """CuTe ``max_common_layout``: the most contiguous elements on which *a* and *b* agree."""
    if get_swizzle_portion(b) is not None and get_swizzle_portion(a) is None:
        return max_common_layout(b, a, major=major)
    swizzle = get_swizzle_portion(a)
    if swizzle is not None:
        common = max_common_layout(a.outer, b, major=major)
        base = 1 << swizzle.base
        return composition(common, base, major=major) if base < size(common) else common
    inverse_b = right_inverse(b, major=major)
    common = coalesce(composition(a, inverse_b, major=major), major=major)
    first = get(common, 0) if isinstance(common.shape, tuple) else common
    if expr.static_dim_value(first.shape) is not None and first.strides == 1:
        return composition(inverse_b, first, major=major)
    return Layout(1, 0)


def max_common_vector(a, b, *, major="row"):
    """CuTe ``max_common_vector``: how many contiguous elements *a* and *b* agree on."""
    swizzle_a, swizzle_b = get_swizzle_portion(a), get_swizzle_portion(b)
    if swizzle_a is not None and swizzle_b is not None:
        vector = max_common_vector(a.outer, b.outer, major=major)
        if swizzle_a == swizzle_b:
            return vector
        return min(vector, 1 << swizzle_a.base, 1 << swizzle_b.base)
    if swizzle_b is not None:
        return max_common_vector(b, a, major=major)
    if swizzle_a is not None:
        return min(max_common_vector(a.outer, b, major=major), 1 << swizzle_a.base)
    common = coalesce(composition(a, right_inverse(b, major=major), major=major), major=major)
    first = get(common, 0) if isinstance(common.shape, tuple) else common
    if expr.static_dim_value(first.shape) is not None and first.strides == 1:
        return first.shape
    return 1


def domain_distribute(a, b: int, *, major="row"):
    """CuTe ``domain_distribute``: an ordered layout of *b* coordinates of shape *a*."""
    if major == "row":
        return reverse_modes(domain_distribute(reverse_modes(a), b, major="col"))
    shapes = flatten_to_tuple(a.shape if isinstance(a, LayoutBase) else a)
    result_shape, rest = [], b
    for extent in shapes:
        common = gcd(extent, rest)
        result_shape.append(common)
        rest //= common
    return coalesce(
        Layout(tuple(result_shape), compact_major(shapes, LayoutLeft)), major="col"
    )


def nullspace(layout, *, major="row"):
    """CuTe ``nullspace``: the coordinates *layout* sends to 0, as a compact layout."""
    if isinstance(layout, ComposedLayout):
        return Layout(1, 0)
    if major == "row":
        return reverse_modes(nullspace(reverse_modes(layout), major="col"))
    steps = flatten_to_tuple(layout.strides)
    zeros = tuple(i for i, d in enumerate(steps) if expr.static_dim_value(d) == 0)
    if not zeros:
        return Layout(1, 0)
    shapes = flatten_to_tuple(layout.shape)
    strides = compact_major(shapes, LayoutLeft)
    return Layout(unwrap(tuple(shapes[i] for i in zeros)), unwrap(tuple(strides[i] for i in zeros)))


def slice(coord, layout):
    """CuTe ``slice``: keep the modes where *coord* is ``None`` (``_``), drop the rest.

    ``layout.hpp``, ``layout_composed.hpp`` and ``underscore.hpp``.
    """
    if isinstance(layout, ComposedLayout):
        return slice_and_offset(coord, layout)[0]
    if isinstance(layout, Layout):
        return Layout(slice(coord, layout.shape), slice(coord, layout.strides))
    if coord is None:
        return layout
    if isinstance(coord, tuple):
        return tuple(
            value
            for c, mode in builtins.zip(coord, layout, strict=True)
            for value in ((mode,) if c is None else slice(c, mode))
        )
    return ()


def dice(coord, layout):
    """CuTe ``dice``: keep the modes where *coord* is given, drop the ``None`` (``_``) ones.

    ``layout.hpp`` and ``underscore.hpp``.
    """
    if isinstance(layout, Layout):
        return Layout(dice(coord, layout.shape), dice(coord, layout.strides))
    if coord is None:
        return ()
    if isinstance(coord, tuple):
        return tuple(
            value
            for c, mode in builtins.zip(coord, layout, strict=True)
            if c is not None
            for value in (dice(c, mode) if isinstance(c, tuple) else (mode,))
        )
    return layout


def all_underscore(coord) -> bool:
    """CuTe ``underscore.hpp`` ``all_underscore``."""
    if isinstance(coord, tuple):
        return all(all_underscore(one) for one in coord)
    return coord is None


def slice_and_offset(coord, layout, *, major="row"):
    """CuTe ``slice_and_offset``: the sliced layout and the index of the fixed coordinates.

    A composed layout moves the index into its own offset (``layout_composed.hpp``).
    A swizzled composed layout with static coordinates and offset follows
    ``swizzle_layout.hpp``. It may return a plain layout, or keep part of the
    index outside the swizzle.
    """
    if isinstance(layout, ComposedLayout):
        values = (layout.offset, *flatten_to_tuple(coord))
        if get_swizzle_portion(layout) is not None and all(
            one is None or isinstance(one, int) for one in values
        ):
            if major == "row":
                sliced, offset = slice_and_offset(
                    reverse_modes(coord),
                    make_composed_layout(layout.inner, layout.offset, reverse_modes(layout.outer)),
                    major="col",
                )
                if isinstance(sliced, ComposedLayout):
                    return make_composed_layout(
                        sliced.inner, sliced.offset, reverse_modes(sliced.outer)
                    ), offset
                return reverse_modes(sliced), offset
            return swizzle_slice_and_offset(coord, layout)
        sliced, offset = slice_and_offset(coord, layout.outer, major=major)
        return make_composed_layout(layout.inner, layout.offset + offset, sliced), 0
    return slice(coord, layout), crd2idx(coord, layout.shape, layout.strides, major=major)


def swizzle_slice_and_offset(coord, layout: ComposedLayout):
    """CuTe ``swizzle_layout.hpp`` ``slice_and_offset`` of a swizzled ``ComposedLayout``."""
    swizzle = layout.inner
    bits, base, shift = swizzle.bits, swizzle.base, abs(swizzle.shift)
    if all_underscore(coord):
        return composition(swizzle, layout.offset, slice(coord, layout.outer)), 0
    if not any(one is None for one in flatten_to_tuple(coord)):
        raise ValueError(
            "slice_and_offset: a swizzled layout needs a coordinate that keeps at least one mode"
        )
    shape = (1 << base, 1 << bits, 1 << (shift - bits), 1 << bits, 1)
    compact = compact_major(shape, LayoutLeft)
    anti_zy = Layout(shape, (compact[0], 0, compact[2], 0, int_tuple.product(shape)))
    only_zy = Layout(shape, (0, compact[1], 0, compact[3], 0))
    sliced = slice(coord, layout.outer)
    diced = dice(coord, layout.outer)
    diced_index = crd2idx(dice(coord, coord), diced.shape, diced.strides, major="col")
    offset_only_zy = layout.offset ^ only_zy(diced_index, major="col")
    offset_anti_zy = anti_zy(diced_index, major="col")
    active = only_zy(sliced(size(sliced) - 1, major="col"), major="col")
    if active & ~swizzle(active):
        return composition(swizzle, offset_only_zy, sliced), offset_anti_zy
    low_z = swizzle.zzz_msk & -swizzle.zzz_msk
    low_y = swizzle.yyy_msk & -swizzle.yyy_msk
    stride_lo = swizzle_strides(low_z < low_y, low_z, low_y, offset_only_zy, bits)
    stride_hi = swizzle_strides(low_z > low_y, low_z, low_y, offset_only_zy, bits)
    twos = unwrap((2,) * bits)
    swizzled = Layout(
        (1 << base, twos, 1 << (shift - bits), twos, 1),
        (1, unwrap(stride_lo), 1 << (base + bits), unwrap(stride_hi), 1 << (base + bits + shift)),
    )
    return (
        composition(swizzled, sliced, major="col"),
        swizzle(offset_only_zy) + offset_anti_zy,
    )


def swizzle_strides(lower, z, y, offset, bits) -> tuple:
    """CuTe ``swizzle_layout.hpp`` ``detail::make_swizzle_strides``."""
    if lower:
        return tuple(z << i if offset & (y << i) == 0 else -(z << i) for i in range(bits))
    return tuple(
        (y + z) << i if offset & (z << i) == 0 else (y - z) << i for i in range(bits)
    )
