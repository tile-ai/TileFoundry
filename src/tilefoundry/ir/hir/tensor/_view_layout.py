"""Common type-inference plumbing for tensor views."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from tilefoundry.ir.types import ComposedLayout, Layout, LayoutBase, TensorType
from tilefoundry.ir.types.stride import try_compact_major


def derive_view_layout(
    source_type: TensorType,
    result_shape: tuple,
    transform: Callable[[Layout], Layout | None],
) -> LayoutBase | None:
    """Apply one view's primitive-layout transform without losing composition."""
    source = source_type.layout
    if source is None:
        source = Layout(tuple(source_type.shape), try_compact_major(tuple(source_type.shape)))
    if isinstance(source, Layout):
        return transform(source) or Layout(result_shape, None)
    if isinstance(source, ComposedLayout) and isinstance(source.outer, Layout):
        outer = transform(source.outer) or Layout(result_shape, None)
        return replace(source, outer=outer)
    return None


__all__ = ["derive_view_layout"]
