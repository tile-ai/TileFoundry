"""Construction and specialization helpers for IR patterns."""

from __future__ import annotations

from tilefoundry.ir.types import ComposedLayout, Swizzle

from .pattern import (
    ComposedLayoutPattern,
    LayoutPattern,
    Pattern,
    RangePattern,
    SwizzlePattern,
)


def arrangement_pattern(
    layout,
    *,
    forward: bool = True,
    injective: bool = True,
    per_mode: bool = False,
):
    """Build the exact pattern for one authored arrangement."""
    if isinstance(layout, ComposedLayout):
        inner = layout.inner
        return ComposedLayoutPattern(
            SwizzlePattern(inner.bits, inner.base, inner.shift)
            if isinstance(inner, Swizzle)
            else inner,
            layout.offset,
            arrangement_pattern(
                layout.outer,
                forward=forward,
                injective=injective,
                per_mode=per_mode,
            ),
        )
    return LayoutPattern(
        tuple(layout.shape),
        tuple(layout.strides),
        forward=forward,
        injective=injective,
        per_mode=per_mode,
    )


def locate_dim_var(params: tuple, name: str) -> tuple[int, int] | None:
    """Return the first parameter/axis carrying a ``DimVar`` named *name*."""
    for index, param in enumerate(params):
        shape = getattr(param.type, "shape", None)
        if shape is None:
            continue
        for axis, dim in enumerate(shape):
            if getattr(dim, "name", None) == name:
                return index, axis
    return None


def _mangle_variant_name(name: str, specializations: tuple[Pattern, ...]) -> str:
    if len(specializations) != 1 or not isinstance(specializations[0], RangePattern):
        raise TypeError("variant requires exactly one RangePattern")
    pattern = specializations[0]
    if not pattern.dim_var or pattern.lo is None or pattern.hi is None:
        raise TypeError("variant RangePattern requires dim_var, lo, and hi")
    return f"{name}${pattern.dim_var}${pattern.lo}_{pattern.hi}"


__all__ = ["_mangle_variant_name", "arrangement_pattern", "locate_dim_var"]
