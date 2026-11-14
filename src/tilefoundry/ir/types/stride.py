"""Strides: the steps a compact arrangement walks, and reading one back.

CuTe keeps this apart from the layout algebra (``stride.hpp``): making the
steps of a compact arrangement, and turning an index back into the coordinate
that reaches it, are not operations on layouts.
"""

from __future__ import annotations

from tilefoundry import services
from tilefoundry.ir import expr

from .int_tuple import flatten

__all__ = [
    "compact_col_major",
    "compact_major",
    "compact_row_major",
    "crd2idx",
    "idx2crd",
    "try_compact_major",
]


def multiply(a, b):
    """Integer product, the default fold of ``compact_major``."""
    return a * b


def compact_major(shape: tuple, *, major: str = "row", mul=None, current=1) -> tuple:
    """CuTe ``compact_major``: the steps a compact arrangement of *shape* walks.

    ``"col"`` makes mode zero the fastest and ``"row"`` the last one, which is
    the one difference between CuTe's two spellings of it. A mode that is
    itself a group of them is walked the same way inside what the modes beside
    it leave it, so the steps come back shaped like the shape they were made
    for. *mul* defaults to integer multiplication; pass a dim-expression fold
    for a shape whose entries are symbolic.
    """
    if not isinstance(shape, tuple):
        return current
    if not shape:
        return ()
    if mul is None:
        mul = multiply
    order = range(len(shape)) if major == "col" else range(len(shape) - 1, -1, -1)
    strides: list = [1] * len(shape)
    acc = current
    for index in order:
        strides[index] = compact_major(shape[index], major=major, mul=mul, current=acc)
        acc = mul(acc, mode_product(shape[index], mul))
    return tuple(strides)


def mode_product(shape, mul):
    """The fold of *mul* over every leaf of a possibly nested *shape*."""
    if not isinstance(shape, tuple):
        return shape
    total = 1
    for one in shape:
        total = mul(total, mode_product(one, mul))
    return total


def compact_col_major(shape: tuple, *, mul=None) -> tuple:
    """CuTe ``compact_col_major``: mode zero walks fastest."""
    return compact_major(shape, major="col", mul=mul)


def compact_row_major(shape: tuple, *, mul=None) -> tuple:
    """CuTe ``compact_row_major``: the last mode walks fastest."""
    return compact_major(shape, major="row", mul=mul)


def try_compact_major(shape: tuple, *, major: str = "row") -> "tuple | None":
    """:func:`compact_major`, or ``None`` where an extent is not a static int."""
    if not all(
        isinstance(one, int) and not isinstance(one, bool)
        for one in flatten(shape)
    ):
        return None
    return compact_major(shape, major=major)


def idx2crd(idx: int, shape: tuple, stride: tuple) -> tuple:
    """CuTe ``idx2crd``: the coordinate in ``<shape, stride>`` an index reaches.

    Per mode ``(idx // stride) % shape``, recursing wherever a mode is itself a
    group of them, so what comes back is shaped like the arrangement it was
    read against. Which mode walks fastest is the strides' to say, not this
    function's.
    """
    return tuple(
        idx2crd(idx, one, step) if isinstance(one, tuple) else 0 if one == 1 else (idx // step) % one
        for one, step in zip(shape, stride)
    )


def crd2idx(crd, shape, stride):
    """CuTe ``crd2idx``, with integer coordinates read in row-major order.

    Decompose from the innermost mode; the final, outermost digit is not
    reduced modulo its extent. Static coordinates keep integer arithmetic.
    """
    if crd is None:
        return 0
    if isinstance(crd, int) and isinstance(shape, int) and isinstance(stride, int):
        return crd * stride
    if isinstance(crd, tuple):
        return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride, strict=True))
    literal = expr.static_dim_value(crd)
    crd = literal if literal is not None else crd
    if not isinstance(shape, tuple):
        step = expr.static_dim_value(stride)
        if literal is not None and step is not None:
            return literal * step
        return services.simplify_dim("mul", (crd, stride))
    if not shape:
        return 0
    if len(shape) == 1:
        return crd2idx(crd, shape[0], stride[0])
    if literal == 0:
        return crd2idx((0,) * len(shape), shape, stride)
    extent = mode_product(shape[-1], multiply)
    if isinstance(crd, int) and isinstance(extent, int):
        quotient, remainder = divmod(crd, extent)
    else:
        quotient = services.simplify_dim("floordiv", (crd, extent))
        remainder = services.simplify_dim("mod", (crd, extent))
    return crd2idx(quotient, shape[:-1], stride[:-1]) + crd2idx(remainder, shape[-1], stride[-1])
