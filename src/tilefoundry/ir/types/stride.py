"""Strides: the steps a compact arrangement walks, and reading one back.

CuTe keeps this apart from the layout algebra (``stride.hpp``): making the
steps of a compact arrangement, and turning an index back into the coordinate
that reaches it, are not operations on layouts.
"""

from __future__ import annotations


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
        mul = lambda a, b: a * b  # noqa: E731
    order = range(len(shape)) if major == "col" else range(len(shape) - 1, -1, -1)
    strides: list = [1] * len(shape)
    acc = current
    for index in order:
        strides[index] = compact_major(shape[index], major=major, mul=mul, current=acc)
        acc = mul(acc, _product(shape[index], mul))
    return tuple(strides)


def _product(shape, mul):
    if not isinstance(shape, tuple):
        return shape
    total = 1
    for one in shape:
        total = mul(total, _product(one, mul))
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
        for one in _flat(shape)
    ):
        return None
    return compact_major(shape, major=major)


def _flat(shape):
    if not isinstance(shape, tuple):
        return (shape,)
    return tuple(one for item in shape for one in _flat(item))


def idx2crd(idx: int, shape: tuple, stride: tuple) -> tuple:
    """CuTe ``idx2crd``: the coordinate in ``<shape, stride>`` an index reaches.

    Per mode ``(idx // stride) % shape``, recursing wherever a mode is itself a
    group of them, so what comes back is shaped like the arrangement it was
    read against. Which mode walks fastest is the strides' to say, not this
    function's.
    """
    return tuple(
        idx2crd(idx, one, step) if isinstance(one, tuple) else (idx // step) % one
        for one, step in zip(shape, stride)
    )


def crd2idx(crd: tuple, shape: tuple, stride: tuple) -> int:
    """CuTe ``crd2idx``: the index a coordinate in ``<shape, stride>`` reaches."""
    total = 0
    for value, one, step in zip(crd, shape, stride):
        total += crd2idx(value, one, step) if isinstance(one, tuple) else value * step
    return total


__all__ = [
    "compact_col_major",
    "compact_major",
    "compact_row_major",
    "crd2idx",
    "idx2crd",
    "try_compact_major",
]
