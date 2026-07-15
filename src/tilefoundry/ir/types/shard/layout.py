from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from tilefoundry.ir.types.shape_dim import ShapeDim


@dataclass(frozen=True)
class Layout:
    """Layout: shape + per-axis strides."""

    shape: tuple["ShapeDim | None", ...]
    strides: Optional[tuple["ShapeDim", ...]] = None


@dataclass(frozen=True)
class ComposedLayout:
    """Composed layout: ``image(c) = inner(offset + outer(c))``.

    Field order + names mirror the reference composed-layout constructor
    ``make_composed_layout(inner, offset, outer)`` (``third_party/cutlass``,
    composed-layout core):

    - ``outer`` — applied **first** (domain / input side); the domain shape and
      axis numbering of the composition come from ``outer``, so a binding
      ``ShardLayout``'s ``Split(k)`` references ``outer``'s domain axis.
    - ``offset`` — intermediate scalar offset added before ``inner``.
    - ``inner`` — applied **last** (codomain / output side).

    The left inverse reverses the composition (see the reference
    ``layout_composed.hpp`` ``left_inverse``):
    ``image⁻¹(t) = outer⁻¹(inner⁻¹(t) − offset)``.
    """

    inner: "LayoutLike"
    offset: int
    outer: "LayoutLike"


# Forward ref resolved after shard_layout import
LayoutLike = Union[Layout, ComposedLayout, "ShardLayout"]  # noqa: F821

EMPTY_LAYOUT = Layout(shape=(), strides=())


__all__ = ["Layout", "ComposedLayout", "LayoutLike", "EMPTY_LAYOUT"]
