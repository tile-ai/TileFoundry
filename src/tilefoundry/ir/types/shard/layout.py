from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .int_tuple import flatten


class LayoutBase:
    """Common domain-shape contract for tensor layout descriptors."""

    @property
    def domain_rank(self) -> int:
        return len(flatten(self.shape))


@dataclass(frozen=True)
class Layout(LayoutBase):
    """Cute-style layout: shape + per-axis cute strides."""

    shape: tuple["ShapeDim | None", ...]
    strides: Optional[tuple["ShapeDim", ...]] = None


@dataclass(frozen=True)
class ComposedLayout(LayoutBase):
    """CuTe composed layout: ``image(c) = inner(offset + outer(c))``.

    Field order + names mirror CuTeDSL ``make_composed_layout(inner, offset,
    outer)`` (``third_party/cutlass/python/CuTeDSL/cutlass/cute/core.py``):

    - ``outer`` — applied **first** (domain / input side); the domain shape and
      axis numbering of the composition come from ``outer``, so a binding
      ``ShardLayout``'s ``Split(k)`` references ``outer``'s domain axis.
    - ``offset`` — intermediate scalar offset added before ``inner``.
    - ``inner`` — applied **last** (codomain / output side); ``None`` is
      identity.

    Either component may be a ``ShardLayout`` so a later placement stage can
    preserve an earlier stage's distribution as a nested layout.

    The left inverse reverses the composition (see CuTe
    ``layout_composed.hpp`` ``left_inverse``):
    ``image⁻¹(t) = outer⁻¹(inner⁻¹(t) − offset)``.
    """

    inner: LayoutBase | None
    offset: int
    outer: LayoutBase | None

    @property
    def shape(self) -> tuple:
        domain = self.outer if self.outer is not None else self.inner
        if domain is None:
            return ()
        return domain.shape


EMPTY_LAYOUT = Layout(shape=(), strides=())


__all__ = ["LayoutBase", "Layout", "ComposedLayout", "EMPTY_LAYOUT"]
