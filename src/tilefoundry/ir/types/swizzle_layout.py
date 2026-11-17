"""CuTe ``swizzle_layout.hpp`` and ``swizzle.hpp`` algebra involving a ``Swizzle``.

The ``Swizzle`` value type itself lives in :mod:`tilefoundry.ir.types.layout`,
alongside the other layout value types. ``supports_composition``,
``supports_inverse`` and ``inverse`` select these overloads for the general
algebra, which Python cannot dispatch by type.
"""

from __future__ import annotations

from typing import Callable, Optional

from .layout import ComposedLayout, Layout, Swizzle

__all__ = [
    "composition",
    "get_nonswizzle_portion",
    "get_swizzle_portion",
    "inverse",
    "make_swizzle",
    "supports_composition",
    "supports_inverse",
]


def make_swizzle(y: int, z: int) -> Swizzle:
    """CuTe ``swizzle.hpp`` ``make_swizzle<Y,Z>``: the swizzle XORing the bits *y* onto *z*."""
    bits_y, bits_z = y.bit_count(), z.bit_count()
    if bits_y != bits_z:
        raise NotImplementedError(
            f"make_swizzle: the Y mask {y:#x} holds {bits_y} bits and the Z mask "
            f"{z:#x} holds {bits_z}; only an equal-width pair is a Swizzle"
        )
    trailing_y = (y & -y).bit_length() - 1 if y else 32
    trailing_z = (z & -z).bit_length() - 1 if z else 32
    swizzle = Swizzle(bits_y, min(trailing_y, trailing_z) % 32, trailing_y - trailing_z)
    if swizzle.swizzle_code != (y | z):
        raise NotImplementedError(
            f"make_swizzle: the mask pair ({y:#x}, {z:#x}) is not a Swizzle<B,M,S>; "
            f"its bits are not two contiguous equal-width runs"
        )
    return swizzle


def get_swizzle_portion(layout: object) -> Optional[Swizzle]:
    """CuTe ``get_swizzle_portion``: the ``Swizzle`` of a composed layout, or ``None``.

    CuTe answers ``Swizzle<0,4,3>`` for a layout with no swizzle; this answers ``None``.
    """
    if isinstance(layout, ComposedLayout) and isinstance(layout.inner, Swizzle):
        return layout.inner
    return None


def get_nonswizzle_portion(layout):
    """CuTe ``get_nonswizzle_portion``: the layout under a swizzle; any other layout as it is."""
    if get_swizzle_portion(layout) is not None:
        return layout.outer
    return layout


def supports_composition(left: object, right: object) -> bool:
    """Return whether the operands select a CuTe swizzle composition overload."""
    return (isinstance(left, Swizzle) and isinstance(right, (Layout, Swizzle))) or (
        isinstance(left, Layout) and isinstance(right, Swizzle)
    )


def composition(left, right, *, major="row"):
    """CuTe ``composition`` of a ``Swizzle`` with a ``Layout`` or another ``Swizzle``.

    ``Swizzle`` then ``Layout`` is ``swizzle_layout.hpp`` (a zero-bit swizzle leaves the
    layout as it is). ``Layout`` then ``Swizzle`` maps the swizzle's masks through the
    layout. Two swizzles of one shift merge (``swizzle.hpp``).
    """
    if isinstance(left, Swizzle) and isinstance(right, Layout):
        return right if left.bits == 0 else ComposedLayout(left, 0, right)
    if isinstance(left, Swizzle) and isinstance(right, Swizzle):
        if left.shift != right.shift:
            raise NotImplementedError("composition: only swizzles of the same shift merge")
        return make_swizzle(left.yyy_msk ^ right.yyy_msk, left.zzz_msk ^ right.zzz_msk)
    if isinstance(left, Layout) and isinstance(right, Swizzle):
        active_y = left(right.yyy_msk, major=major)
        active_z = left(right.zzz_msk, major=major)
        return composition(make_swizzle(active_y, active_z), left)
    raise NotImplementedError(
        f"composition: no swizzle rule for {type(left).__name__} ∘ {type(right).__name__}"
    )


def supports_inverse(layout: object) -> bool:
    """Return whether a CuTe swizzle inverse overload accepts *layout*."""
    return isinstance(layout, Swizzle) or get_swizzle_portion(layout) is not None


def inverse(layout, inverse_of_layout: Callable, *, major="row"):
    """CuTe ``left_inverse`` / ``right_inverse`` of a swizzle or a swizzled composed layout.

    A swizzle is its own inverse. With offset 0 the result composes the inverse of
    the outer layout with the swizzle; otherwise it is the composed layout of the
    inverses in reverse order, with the offset negated.
    """
    if isinstance(layout, Swizzle):
        return layout
    if layout.offset == 0:
        return composition(inverse_of_layout(layout.outer), layout.inner, major=major)
    return ComposedLayout(inverse_of_layout(layout.outer), -layout.offset, layout.inner)
