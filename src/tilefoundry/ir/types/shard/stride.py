"""Strides: the steps a compact arrangement walks, and reading one back.

CuTe keeps this apart from the layout algebra (``stride.hpp``): making the
steps of a compact arrangement, and turning an index back into the coordinate
that reaches it, are not operations on layouts.
"""

from __future__ import annotations


def prefix_product(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Exclusive prefix product (column-major natural strides)."""
    out: list[int] = []
    acc = 1
    for s in shape:
        out.append(acc)
        acc *= s
    return tuple(out)


def c_order_strides(shape: tuple, *, mul=None) -> tuple:
    """Row-major (C-order) contiguous strides.

    Row-major (C-order) contiguous strides: ``strides[-1] == 1``,
    ``strides[i] == strides[i+1] * shape[i+1]``.

    The single home for this computation. *mul* defaults to ``int``
    multiplication; pass a dim-expression fold (e.g. wrapping
    ``simplify_dim(DimMul, ...)``) for shapes with symbolic entries.
    """
    if not shape:
        return ()
    if mul is None:
        mul = lambda a, b: a * b  # noqa: E731
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = mul(strides[i + 1], shape[i + 1])
    return tuple(strides)


def try_c_order_strides(shape: tuple) -> tuple[int, ...] | None:
    """``c_order_strides`` when every entry is a static non-bool ``int``, else ``None``.

    ``c_order_strides`` when every entry is a static non-bool ``int``,
    else ``None`` (symbolic / dynamic shapes have no static strides).
    """
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in shape):
        return None
    return c_order_strides(shape)


def idx2crd(idx: int, shape: tuple[int, ...], stride: tuple[int, ...]) -> tuple[int, ...]:
    """Per-mode ``(idx // stride_i) % shape_i`` (CuTe ``idx2crd``)."""
    return tuple((idx // d) % s for s, d in zip(shape, stride))


__all__ = ["c_order_strides", "idx2crd", "prefix_product", "try_c_order_strides"]
