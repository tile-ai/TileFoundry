from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class Layout:
    """Cute-style layout: shape + per-axis cute strides."""

    shape: tuple["ShapeDim | None", ...]
    strides: Optional[tuple["ShapeDim", ...]] = None


@dataclass(frozen=True)
class ComposedLayout:
    """CuTe composed layout: ``image(c) = inner(offset + outer(c))``.

    Field order + names mirror CuTeDSL ``make_composed_layout(inner, offset,
    outer)`` (``third_party/cutlass/python/CuTeDSL/cutlass/cute/core.py``):

    - ``outer`` — applied **first** (domain / input side); the domain shape and
      axis numbering of the composition come from ``outer``, so a binding
      ``ShardLayout``'s ``Split(k)`` references ``outer``'s domain axis.
    - ``offset`` — intermediate scalar offset added before ``inner``.
    - ``inner`` — applied **last** (codomain / output side).

    The left inverse reverses the composition (see CuTe
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
