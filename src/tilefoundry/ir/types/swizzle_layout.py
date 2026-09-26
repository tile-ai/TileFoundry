"""Provide CuTe layout algebra specializations involving a ``Swizzle``.

The operations here mirror ``cute/swizzle_layout.hpp``. The ``Swizzle``
value type itself remains in :mod:`tilefoundry.ir.types.layout`, alongside
the other layout value types.
"""

from __future__ import annotations

from typing import Callable, Optional

from .layout import (
    ComposedLayout,
    Layout,
    Swizzle,
    flat_shape,
    flat_stride,
    make_swizzle,
)


def _crd2idx_unbounded(layout: Layout, coord: int) -> int:
    """Apply CuTe ``crd2idx`` while leaving the final mode unwrapped."""
    shape = flat_shape(layout)
    stride = flat_stride(layout)
    idx = 0
    rem = coord
    last = len(shape) - 1
    for position, (extent, step) in enumerate(zip(shape, stride)):
        if position == last:
            idx += rem * step
        else:
            idx += (rem % extent) * step
            rem //= extent
    return idx


def get_swizzle_portion(layout: object) -> Optional[Swizzle]:
    """Return the final ``Swizzle`` of a composed layout, or ``None``.

    This mirrors CuTe ``get_swizzle_portion`` while using ``None`` rather
    than an identity ``Swizzle<0,4,3>`` for a non-swizzled layout.
    """
    if isinstance(layout, ComposedLayout) and isinstance(layout.inner, Swizzle):
        return layout.inner
    return None


def supports_composition(left: object, right: object) -> bool:
    """Return whether the operands select a CuTe swizzle composition overload."""
    return (isinstance(left, Swizzle) and isinstance(right, Layout)) or (
        isinstance(left, Layout) and isinstance(right, Swizzle)
    )


def composition(left, right, offset: int = 0):
    """CuTe ``composition`` overloads involving a ``Swizzle``."""
    if isinstance(left, Swizzle) and isinstance(right, Layout):
        if left.bits == 0 and offset == 0:
            return right
        return ComposedLayout(inner=left, offset=offset, outer=right)
    if isinstance(left, Layout) and isinstance(right, Swizzle):
        if offset:
            raise NotImplementedError(
                f"composition: a non-zero offset ({offset}) between a Layout and a "
                f"Swizzle has no canonical ComposedLayout form"
            )
        active_y = _crd2idx_unbounded(left, right.yyy_mask)
        active_z = _crd2idx_unbounded(left, right.zzz_mask)
        return composition(make_swizzle(active_y, active_z), left)
    raise NotImplementedError(
        f"composition: no swizzle rule for {type(left).__name__} ∘ {type(right).__name__}"
    )


def supports_inverse(layout: object) -> bool:
    """Return whether a CuTe swizzle inverse overload accepts *layout*."""
    return isinstance(layout, Swizzle) or get_swizzle_portion(layout) is not None


def inverse(layout, inverse_of_layout: Callable):
    """Apply the CuTe swizzle inverse using the injected general inverse."""
    if isinstance(layout, Swizzle):
        return layout
    if layout.offset != 0:
        raise NotImplementedError(
            f"inverse: a swizzled composed layout with a non-zero offset "
            f"({layout.offset}) inverts to a Swizzle on the domain side, which "
            f"ComposedLayout.outer cannot hold"
        )
    if not isinstance(layout.outer, Layout):
        raise NotImplementedError(
            f"inverse: a swizzled composed layout inverts through its outer "
            f"Layout; this one states {type(layout.outer).__name__}"
        )
    return composition(inverse_of_layout(layout.outer), layout.inner)


__all__ = [
    "composition",
    "get_swizzle_portion",
    "inverse",
    "supports_composition",
    "supports_inverse",
]
