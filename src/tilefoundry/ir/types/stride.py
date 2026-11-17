"""The CuTe ``stride.hpp`` functions: coordinates, indices and compact strides.

CuTe keeps this apart from the layout algebra: making the steps of a compact
layout, and turning a coordinate into an index and back, are not operations on
layouts. ``major="row"`` reverses every level of the inputs, calls the CuTe
body with ``major="col"`` and reverses a coordinate result back.
"""

from __future__ import annotations

from tilefoundry import services
from tilefoundry.ir import expr

from .int_tuple import (
    flatten,
    flatten_to_tuple,
    product,
    product_like,
    reverse_modes,
    unflatten,
)

__all__ = [
    "LayoutLeft",
    "LayoutRight",
    "compact_col_major",
    "compact_major",
    "compact_order",
    "compact_row_major",
    "crd2crd",
    "crd2idx",
    "idx2crd",
]

LayoutLeft = "col"
LayoutRight = "row"


def crd2idx(coord, shape, stride=None, *, major="row"):
    """CuTe ``stride.hpp`` ``crd2idx(c, s, d)``, and ``crd2idx(c, s)`` when *stride* is None.

    An integer coordinate of a tuple shape divides from mode 0 and leaves the last
    mode unreduced. ``None`` in a coordinate is CuTe's ``_`` and reads as 0. A
    ``Layout`` in place of *shape* supplies its own strides (``layout.hpp``).
    """
    if stride is None and hasattr(shape, "strides"):
        shape, stride = shape.shape, shape.strides
    if major == "row":
        return crd2idx(
            reverse_modes(coord), reverse_modes(shape), reverse_modes(stride), major="col"
        )
    if coord is None:
        return 0
    if stride is None:
        if not isinstance(coord, tuple):
            return coord
        flat_coord, shapes = flatten(coord), flatten(product_like(shape, coord))
        index = flat_coord[-1]
        for c, s in zip(reversed(flat_coord[:-1]), reversed(shapes[:-1])):
            index = c + s * index
        return index
    if isinstance(coord, tuple):
        if not isinstance(shape, tuple):
            raise ValueError(f"crd2idx: a tuple coordinate {coord!r} of a scalar shape")
        return sum(
            crd2idx(c, s, d, major="col") for c, s, d in zip(coord, shape, stride, strict=True)
        )
    literal = expr.static_dim_value(coord)
    coord = literal if literal is not None else coord
    if not isinstance(shape, tuple):
        step = expr.static_dim_value(stride)
        if literal == 0 or (literal is not None and step is not None):
            return literal * (step or 0)
        return services.simplify_dim("mul", (coord, stride))
    if len(shape) != len(stride):
        raise ValueError(f"crd2idx: mismatched ranks {len(shape)} and {len(stride)}")
    if not shape:
        return 0
    if len(shape) == 1:
        return crd2idx(coord, shape[0], stride[0], major="col")
    if literal == 0:
        return sum(crd2idx(0, s, d, major="col") for s, d in zip(shape, stride))
    extent = product(shape[0])
    if isinstance(coord, int) and isinstance(extent, int):
        quotient, remainder = divmod(coord, extent)
    else:
        quotient = services.simplify_dim("floordiv", (coord, extent))
        remainder = services.simplify_dim("mod", (coord, extent))
    return crd2idx(remainder, shape[0], stride[0], major="col") + crd2idx(
        quotient, shape[1:], stride[1:], major="col"
    )


def idx2crd(idx, shape, stride=None, *, major="row"):
    """CuTe ``stride.hpp`` ``idx2crd(i, s, d)``, and ``idx2crd(i, s)`` when *stride* is None.

    Per mode ``(idx / d) % s``; a static extent of 1 gives 0. An integer *stride* of
    a tuple shape starts ``compact_col_major``. The two-argument form divides from
    mode 0 and leaves the last mode unreduced.
    """
    if major == "row":
        return reverse_modes(
            idx2crd(reverse_modes(idx), reverse_modes(shape), reverse_modes(stride), major="col")
        )
    if isinstance(idx, tuple):
        if stride is None:
            return tuple(idx2crd(i, s, major="col") for i, s in zip(idx, shape, strict=True))
        return tuple(
            idx2crd(i, s, d, major="col") for i, s, d in zip(idx, shape, stride, strict=True)
        )
    if stride is None:
        if not isinstance(shape, tuple):
            return idx
        result = []
        for s in shape[:-1]:
            extent = product(s)
            result.append(idx2crd(idx % extent, s, major="col"))
            idx = idx // extent
        return (*result, idx2crd(idx, shape[-1], major="col"))
    if isinstance(shape, tuple):
        if not isinstance(stride, tuple):
            stride = compact_major(shape, "col", stride)
        return tuple(idx2crd(idx, s, d, major="col") for s, d in zip(shape, stride, strict=True))
    if expr.static_dim_value(shape) == 1:
        return 0
    return (idx // stride) % shape


def crd2crd(coord, src_shape, dst_shape, *, major="row"):
    """CuTe ``stride.hpp`` ``crd2crd``: a coordinate of *src_shape* as one of *dst_shape*."""
    if major == "row":
        return reverse_modes(
            crd2crd(
                reverse_modes(coord), reverse_modes(src_shape), reverse_modes(dst_shape), major="col"
            )
        )
    if all(isinstance(value, tuple) for value in (coord, src_shape, dst_shape)):
        return tuple(
            crd2crd(c, s, d, major="col")
            for c, s, d in zip(coord, src_shape, dst_shape, strict=True)
        )
    return idx2crd(crd2idx(coord, src_shape, major="col"), dst_shape, major="col")


def compact(shape, current):
    """CuTe ``stride.hpp`` ``detail::compact<LayoutLeft>``: the strides and the next stride.

    A product with a symbolic extent is a dimension expression.
    """
    if isinstance(shape, tuple):
        strides = []
        for mode in shape:
            stride, current = compact(mode, current)
            strides.append(stride)
        return tuple(strides), current
    if expr.static_dim_value(shape) == 1:
        return 0, current
    if isinstance(current, int) and isinstance(shape, int):
        return current, current * shape
    return current, services.simplify_dim("mul", (current, shape))


def compact_major(shape, major="row", current=1):
    """CuTe ``stride.hpp`` ``compact_major<Major>``: the strides of a compact layout of *shape*.

    ``major`` is ``LayoutLeft`` (``"col"``, mode 0 fastest) or ``LayoutRight``
    (``"row"``, the last mode fastest). A static extent of 1 gets stride 0. A tuple
    *current* gives each mode its own start.
    """
    if major == "row":
        return reverse_modes(compact_major(reverse_modes(shape), "col", reverse_modes(current)))
    if isinstance(current, tuple):
        return tuple(compact_major(s, "col", c) for s, c in zip(shape, current, strict=True))
    return compact(shape, current)[0]


def compact_col_major(shape, current=1):
    """CuTe ``stride.hpp`` ``compact_col_major``: mode 0 walks fastest."""
    return compact_major(shape, LayoutLeft, current)


def compact_row_major(shape, current=1):
    """CuTe ``stride.hpp`` ``compact_row_major``: the last mode walks fastest."""
    return compact_major(shape, LayoutRight, current)


def compact_order(shape, order):
    """CuTe ``stride.hpp`` ``compact_order``: compact strides that follow the ranks in *order*.

    *order* may also be ``LayoutLeft`` or ``LayoutRight``. A symbolic entry of
    *order* ranks above every static one, left to right.
    """
    if isinstance(order, str):
        return compact_major(shape, order)
    ref_shape = flatten_to_tuple(product_like(shape, order))
    flat_order = flatten_to_tuple(order)
    statics = [expr.static_dim_value(one) for one in flat_order]
    max_order = 0
    for one in statics:
        if one is not None and max_order < one:
            max_order = one
    ref_order = tuple(
        one if one is not None else max_order + 1 + index for index, one in enumerate(statics)
    )
    return order_strides(shape, unflatten(ref_order, order), ref_shape, ref_order)


def order_strides(shape, order, ref_shape, ref_order):
    """CuTe ``stride.hpp`` ``detail::compact_order``: a leaf starts past every smaller one."""
    if isinstance(order, tuple):
        return tuple(
            order_strides(s, o, ref_shape, ref_order) for s, o in zip(shape, order, strict=True)
        )
    start = product(tuple(s if o < order else 1 for s, o in zip(ref_shape, ref_order)))
    return compact_col_major(shape, start)
