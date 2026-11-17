"""The CuTe layout types and their ``layout.hpp`` / ``layout_composed.hpp`` construction and access.

``Layout`` is CuTe ``Layout<Shape, Stride>``; its shape and strides may be a
scalar. ``ComposedLayout`` is CuTe ``ComposedLayout<A, Offset, B>`` with ``inner``
for ``layout_a()`` and ``outer`` for ``layout_b()``. ``Swizzle`` is CuTe
``Swizzle<B, M, S>`` (``swizzle.hpp``). Functions taking a layout also take an
IntTuple and then read it as ``int_tuple`` does.
"""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir import expr

from . import int_tuple
from .stride import LayoutRight, compact_major, crd2idx

__all__ = [
    "ComposedLayout",
    "Layout",
    "LayoutBase",
    "NotProjectableError",
    "Swizzle",
    "canonical_mode",
    "coshape",
    "cosize",
    "depth",
    "domain_offset",
    "flatten",
    "get",
    "plain_layout",
    "rank",
    "select",
    "size",
    "take",
    "unflatten",
]


class LayoutBase:
    """Common domain-shape contract for tensor layout descriptors."""


class NotProjectableError(ValueError):
    """A layout cannot serve as a mesh execution scope (not inverse-projectable)."""


@dataclass(frozen=True)
class Layout(LayoutBase):
    """CuTe ``layout.hpp`` ``Layout<Shape, Stride>``; a scalar shape is one mode.

    Without strides the layout is compact row-major (CuTe ``LayoutRight``), and a
    static extent of 1 gets stride 0. Calling it is CuTe ``operator()``: ``crd2idx``.
    """

    shape: "int_tuple.IntTuple"
    strides: "int_tuple.IntTuple" = None

    def __post_init__(self):
        """Fill in compact row-major strides when none are stated."""
        if self.strides is not None:
            return
        shape = canonical_mode(tuple(self.shape) if isinstance(self.shape, list) else self.shape)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "strides", compact_major(shape, LayoutRight))

    def __call__(self, coord, *, major="row"):
        """CuTe ``Layout::operator()``: the index of *coord*."""
        return crd2idx(coord, self.shape, self.strides, major=major)


def canonical_mode(mode):
    """*mode* with every integer ``Constant`` replaced by its plain integer."""
    if isinstance(mode, tuple):
        return tuple(canonical_mode(child) for child in mode)
    if isinstance(mode, int):
        return mode
    return mode.value if isinstance(mode, expr.Constant) and isinstance(mode.value, int) else mode


@dataclass(frozen=True)
class Swizzle:
    """CuTe ``swizzle.hpp`` ``Swizzle<B,M,S>``: the XOR offset functor, not a layout.

    It permutes an index and has no domain shape of its own, so it is not a
    ``LayoutBase``. ``bits`` is ``BBits``, ``base`` is ``MBase`` and ``shift`` is
    ``SShift``, which is also ``msk_sft``. See [shard §4.1](docs/spec/shard.md#41-swizzle).
    """

    bits: int
    base: int
    shift: int

    def __post_init__(self) -> None:
        """Refuse a swizzle that is not a ``Swizzle<B,M,S>``."""
        for name in ("bits", "base", "shift"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Swizzle {name} must be an int, got {value!r}")
        if self.bits < 0:
            raise ValueError(f"Swizzle bits must be non-negative, got {self.bits}")
        if self.base < 0:
            raise ValueError(f"Swizzle base must be non-negative, got {self.base}")
        if abs(self.shift) < self.bits:
            raise ValueError(
                f"abs(Swizzle shift) must be >= bits, got shift={self.shift}, bits={self.bits}"
            )

    @property
    def bit_msk(self) -> int:
        """CuTe ``bit_msk``: the ``bits`` low bits."""
        return (1 << self.bits) - 1

    @property
    def yyy_msk(self) -> int:
        """CuTe ``yyy_msk``: the bits read out of the index."""
        return self.bit_msk << (self.base + max(0, self.shift))

    @property
    def zzz_msk(self) -> int:
        """CuTe ``zzz_msk``: the bits XORed in the index."""
        return self.bit_msk << (self.base - min(0, self.shift))

    @property
    def swizzle_code(self) -> int:
        """CuTe ``swizzle_code``: every bit this swizzle touches."""
        return self.yyy_msk | self.zzz_msk

    def __call__(self, offset: int) -> int:
        """CuTe ``Swizzle::apply``: ``offset ^ shiftr(offset & yyy_msk, msk_sft)``."""
        selected = offset & self.yyy_msk
        moved = selected >> self.shift if self.shift >= 0 else selected << -self.shift
        return offset ^ moved


@dataclass(frozen=True)
class ComposedLayout(LayoutBase):
    """CuTe ``layout_composed.hpp`` ``ComposedLayout``: ``inner(offset + outer(c))``.

    ``inner`` is ``layout_a()``, ``offset`` is ``offset()`` and ``outer`` is
    ``layout_b()``, which gives the domain. ``None`` is an identity component.
    ``inner`` may be a :class:`Swizzle`. ``outer`` and a ``LayoutBase`` ``inner``
    may retain a nested ``ShardLayout``. See [shard §4](docs/spec/shard.md#4-composedlayout).
    """

    inner: "LayoutBase | Swizzle | None"
    offset: int
    outer: LayoutBase | None

    @property
    def shape(self):
        """CuTe ``shape()``: the outer layout's shape, else the inner one's."""
        domain = self.outer if self.outer is not None else self.inner
        if domain is None or isinstance(domain, Swizzle):
            return ()
        return domain.shape

    def __call__(self, coord, *, major="row"):
        """CuTe ``ComposedLayout::operator()``: ``inner(offset + outer(coord))``."""
        index = self.offset + (coord if self.outer is None else self.outer(coord, major=major))
        if self.inner is None:
            return index
        if isinstance(self.inner, Swizzle):
            return self.inner(index)
        return self.inner(index, major=major)


def get(layout, *indices):
    """CuTe ``get<I...>``: the mode at *indices*; ``layout.hpp`` and ``layout_composed.hpp``."""
    if isinstance(layout, ComposedLayout):
        return ComposedLayout(layout.inner, layout.offset, get(layout.outer, *indices))
    if isinstance(layout, Layout):
        return Layout(int_tuple.get(layout.shape, *indices), int_tuple.get(layout.strides, *indices))
    return int_tuple.get(layout, *indices)


def take(layout, b, e):
    """CuTe ``take<B,E>``: the modes in ``[b, e)``; ``layout.hpp`` and ``layout_composed.hpp``."""
    if isinstance(layout, ComposedLayout):
        return ComposedLayout(layout.inner, layout.offset, take(layout.outer, b, e))
    if isinstance(layout, Layout):
        return Layout(int_tuple.take(layout.shape, b, e), int_tuple.take(layout.strides, b, e))
    return int_tuple.take(layout, b, e)


def select(layout, *indices):
    """CuTe ``select<I...>``: the modes at *indices*; ``layout.hpp``."""
    if isinstance(layout, ComposedLayout):
        return ComposedLayout(layout.inner, layout.offset, select(layout.outer, *indices))
    if isinstance(layout, Layout):
        return Layout(int_tuple.select(layout.shape, *indices), int_tuple.select(layout.strides, *indices))
    return int_tuple.select(layout, *indices)


def flatten(layout):
    """CuTe ``flatten``: depth at most one; ``layout.hpp`` and ``layout_composed.hpp``.

    A ``ComposedLayout`` keeps its inner layout and offset.
    """
    if isinstance(layout, ComposedLayout):
        outer = None if layout.outer is None else flatten(layout.outer)
        return ComposedLayout(layout.inner, layout.offset, outer)
    if isinstance(layout, Layout):
        return Layout(int_tuple.flatten(layout.shape), int_tuple.flatten(layout.strides))
    return int_tuple.flatten(layout)


def unflatten(layout, profile):
    """CuTe ``unflatten``: a flat layout or tuple nested to *profile*'s structure."""
    if isinstance(layout, Layout):
        return Layout(
            int_tuple.unflatten(layout.shape, profile),
            int_tuple.unflatten(layout.strides, profile),
        )
    return int_tuple.unflatten(layout, profile)


def rank(layout, *indices):
    """CuTe ``rank``: the number of top-level modes; ``layout.hpp`` and ``layout_composed.hpp``."""
    if isinstance(layout, LayoutBase):
        return int_tuple.rank(layout.shape, *indices)
    return int_tuple.rank(layout, *indices)


def depth(layout, *indices):
    """CuTe ``depth``: how deeply the shape nests; ``layout.hpp`` and ``layout_composed.hpp``."""
    if isinstance(layout, LayoutBase):
        return int_tuple.depth(layout.shape, *indices)
    return int_tuple.depth(layout, *indices)


def size(layout, *indices):
    """CuTe ``size``: the number of coordinates of the mode at *indices*."""
    if isinstance(layout, LayoutBase):
        return int_tuple.size(layout.shape, *indices)
    return int_tuple.size(layout, *indices)


def coshape(layout: Layout, *indices):
    """CuTe ``layout.hpp`` ``coshape``: one past the largest index of the mode at *indices*.

    A symbolic stride counts as non-negative.
    """
    shape = int_tuple.get(layout.shape, *indices)
    strides = int_tuple.get(layout.strides, *indices)
    last = int_tuple.inner_product(
        int_tuple.transform_leaf(shape, lambda s: s - 1),
        int_tuple.transform_leaf(strides, lambda d: abs(d) if isinstance(d, int) else d),
    )
    return last + 1


def cosize(layout, *indices):
    """CuTe ``cosize``: the codomain size; ``layout.hpp`` and ``layout_composed.hpp``.

    A composed layout's codomain is its outer layout's.
    """
    if isinstance(layout, ComposedLayout):
        return cosize(layout.outer, *indices)
    return int_tuple.size(coshape(layout, *indices))


def domain_offset(coord, layout, *, major="row"):
    """CuTe ``domain_offset``: the layout to use at *coord* and the index left to add.

    A plain layout returns ``(layout, layout(coord))`` (``layout.hpp``). A composed
    layout moves its own offset and returns ``(layout, 0)`` (``layout_composed.hpp``).
    """
    if isinstance(layout, ComposedLayout):
        moved = coord if layout.outer is None else layout.outer(coord, major=major)
        return ComposedLayout(layout.inner, layout.offset + moved, layout.outer), 0
    return layout, layout(coord, major=major)


def plain_layout(layout):
    """CuTe ``ComposedLayout::layout_b()``; a plain layout is returned as it is."""
    return layout.outer if isinstance(layout, ComposedLayout) else layout

