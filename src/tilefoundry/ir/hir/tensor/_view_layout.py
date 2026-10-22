"""Common type-inference plumbing for tensor views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tilefoundry.ir.types import ComposedLayout, Layout, LayoutBase, TensorType


def derive_view_layout(
    source_type: TensorType,
    result_shape: tuple,
    transform: Callable[[Layout], Layout | None],
) -> LayoutBase | None:
    """Apply one view's primitive-layout transform without losing composition."""
    source = source_type.layout
    if isinstance(source, Layout):
        return transform(source) or Layout(result_shape)
    if isinstance(source, ComposedLayout) and isinstance(source.outer, Layout):
        outer = transform(source.outer) or Layout(result_shape)
        return replace(source, outer=outer)
    return None


__all__ = ["derive_view_layout"]
