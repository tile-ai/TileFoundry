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
    """Represent ``image(c) = inner(offset + outer(c))``.

    ``outer`` defines the domain shape and axis numbering; ``None`` means an
    identity component. Either component may retain a nested ``ShardLayout``.
    Inversion applies the component inverses in reverse order.

    See [shard §4](docs/spec/shard.md#4-composedlayout).
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
