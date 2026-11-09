"""Common type-inference plumbing for tensor views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tilefoundry.ir import types


def derive_view_layout(
    source_type: types.TensorType,
    result_shape: tuple,
    transform: Callable[[types.Layout], types.Layout | None],
) -> types.LayoutBase | None:
    """Apply one view's primitive-layout transform without losing composition."""
    source = source_type.layout
    if isinstance(source, types.Layout):
        return transform(source) or types.Layout(result_shape)
    if isinstance(source, types.ComposedLayout) and isinstance(source.outer, types.Layout):
        outer = transform(source.outer) or types.Layout(result_shape)
        return replace(source, outer=outer)
    return None


__all__ = ["derive_view_layout"]
