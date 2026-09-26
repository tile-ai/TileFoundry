"""Provide CuTe layout algebra specializations involving a ``Swizzle``.

The operations here mirror ``cute/swizzle_layout.hpp``. The ``Swizzle``
value type itself remains in :mod:`tilefoundry.ir.types.layout`, alongside
the other layout value types.
"""

from __future__ import annotations

from typing import Optional

from .int_tuple import flatten
from .layout import ComposedLayout, Layout, Swizzle
from .stride import compact_col_major


def _shape(layout: Layout) -> tuple[int, ...]:
    return flatten(layout.shape)


def _stride(layout: Layout) -> tuple[int, ...]:
    if layout.strides is not None:
        return layout.strides
    return compact_col_major(_shape(layout))


def _crd2idx_unbounded(layout: Layout, coord: int) -> int:
    """Apply CuTe ``crd2idx`` while leaving the final mode unwrapped."""
    shape = _shape(layout)
    stride = _stride(layout)
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


def make_swizzle(active_y: int, active_z: int) -> Swizzle:
    """CuTe ``make_swizzle<Y,Z>()``: build the swizzle XORing *Y* onto *Z*."""
    bits_y, bits_z = active_y.bit_count(), active_z.bit_count()
    if bits_y != bits_z:
        raise NotImplementedError(
            f"composition: the Y mask {active_y:#x} holds {bits_y} bits and the Z "
            f"mask {active_z:#x} holds {bits_z}; only an equal-width pair is a "
            f"Swizzle"
        )
    if bits_y == 0:
        return Swizzle(0, 0, 0)
    trailing_y = (active_y & -active_y).bit_length() - 1
    trailing_z = (active_z & -active_z).bit_length() - 1
    swizzle = Swizzle(bits_y, min(trailing_y, trailing_z), trailing_y - trailing_z)
    if swizzle.swizzle_code != (active_y | active_z):
        raise NotImplementedError(
            f"composition: the mask pair ({active_y:#x}, {active_z:#x}) is not a "
            f"Swizzle<B,M,S>; its bits are not two contiguous equal-width runs"
        )
    return swizzle


def _supports_composition(left: object, right: object) -> bool:
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


def _supports_inverse(layout: object) -> bool:
    return isinstance(layout, Swizzle) or get_swizzle_portion(layout) is not None


def _inverse(layout, inverse_of_layout):
    """CuTe's composed swizzle inverse, shared by both inverse overloads."""
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


def left_inverse(layout):
    """CuTe's swizzled ``left_inverse`` overload."""
    from .layout_algebra import left_inverse as inverse_of_layout  # noqa: PLC0415

    return _inverse(layout, inverse_of_layout)


def right_inverse(layout):
    """CuTe's swizzled ``right_inverse`` overload."""
    from .layout_algebra import right_inverse as inverse_of_layout  # noqa: PLC0415

    return _inverse(layout, inverse_of_layout)


__all__ = [
    "composition",
    "get_swizzle_portion",
    "left_inverse",
    "make_swizzle",
    "right_inverse",
]
