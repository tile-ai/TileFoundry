from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .int_tuple import flatten as _flat
from .int_tuple import product


class LayoutBase:
    """Common domain-shape contract for tensor layout descriptors."""

    @property
    def domain_rank(self) -> int:
        return len(flatten(self.shape))


@dataclass(frozen=True)
class Layout(LayoutBase):
    """Cute-style layout: shape + per-axis cute strides."""

    shape: tuple["ShapeDim", ...]
    strides: Optional[tuple["ShapeDim", ...]] = None


@dataclass(frozen=True)
class Swizzle:
    """CuTe ``Swizzle<B,M,S>``: the XOR offset functor, not a layout.

    It states a permutation of an *index*, so it has no domain shape of its
    own and is not a ``LayoutBase``. The domain comes from the
    ``ComposedLayout`` that carries it in ``inner``.

    ``bits`` is CuTe's ``BBits`` (how many bits are XORed), ``base`` its
    ``MBase`` (how many low bits are left alone) and ``shift`` its ``SShift``
    (the signed distance from the source mask to the target mask).

    See [shard §4.1](docs/spec/shard.md#41-swizzle).
    """

    bits: int
    base: int
    shift: int

    def __post_init__(self) -> None:
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
                f"abs(Swizzle shift) must be >= bits, got shift={self.shift}, "
                f"bits={self.bits}"
            )

    @property
    def bit_mask(self) -> int:
        """CuTe ``bit_msk``: the ``bits`` low bits."""
        return (1 << self.bits) - 1

    @property
    def yyy_mask(self) -> int:
        """CuTe ``yyy_msk``: the bits read out of the index."""
        return self.bit_mask << (self.base + max(0, self.shift))

    @property
    def zzz_mask(self) -> int:
        """CuTe ``zzz_msk``: the bits XORed in the index."""
        return self.bit_mask << (self.base - min(0, self.shift))

    @property
    def swizzle_code(self) -> int:
        """CuTe ``swizzle_code``: every bit this swizzle touches."""
        return self.yyy_mask | self.zzz_mask

    def __call__(self, offset: int) -> int:
        """``offset ^ shiftr(offset & yyy_msk, msk_sft)`` (CuTe ``apply``)."""
        selected = offset & self.yyy_mask
        moved = selected >> self.shift if self.shift >= 0 else selected << -self.shift
        return offset ^ moved


@dataclass(frozen=True)
class ComposedLayout(LayoutBase):
    """Represent ``image(c) = inner(offset + outer(c))``.

    ``outer`` defines the domain shape and axis numbering; ``None`` means an
    identity component. ``outer`` and a ``LayoutBase`` ``inner`` may retain a
    nested ``ShardLayout``. Inversion applies the component inverses in reverse
    order. ``inner`` may also be a :class:`Swizzle`, which states a
    non-affine permutation of the index and carries no domain of its own.

    See [shard §4](docs/spec/shard.md#4-composedlayout).
    """

    inner: "LayoutBase | Swizzle | None"
    offset: int
    outer: LayoutBase | None

    @property
    def shape(self) -> tuple:
        domain = self.outer if self.outer is not None else self.inner
        if domain is None or isinstance(domain, Swizzle):
            return ()
        return domain.shape


EMPTY_LAYOUT = Layout(shape=(), strides=())


def size(layout: Layout) -> int:
    return product(layout.shape)


def flatten(layout):
    """CuTe ``flatten``: every mode at the top level, of an arrangement or a tuple.

    CuTe spells this once for each (``layout.hpp`` and the tuple algorithms);
    here one name reads both, because which was handed over is plain from what
    comes back.
    """
    if not isinstance(layout, LayoutBase):
        return _flat(layout)
    if isinstance(layout, ComposedLayout):
        return flatten(layout.outer) if layout.outer is not None else EMPTY_LAYOUT
    strides = getattr(layout, "strides", None)
    return Layout(
        shape=_flat(layout.shape),
        strides=None if strides is None else _flat(strides),
    )


def unflatten(layout: LayoutBase, profile) -> "Layout":
    """CuTe ``unflatten``: a flat arrangement nested to *profile*'s shape."""
    from .int_tuple import unflatten as unflatten_tuple  # noqa: PLC0415 - cycle guard

    strides = getattr(layout, "strides", None)
    return Layout(
        shape=unflatten_tuple(tuple(layout.shape), profile),
        strides=None if strides is None else unflatten_tuple(tuple(strides), profile),
    )


def rank(layout: LayoutBase) -> int:
    """CuTe ``rank``: how many modes a layout states at its top level."""
    return len(layout.shape)


def get(layout: LayoutBase, index: int) -> "Layout":
    """CuTe ``get<I>``: one mode of a layout, as a layout of its own."""
    shape = layout.shape[index]
    strides = layout.strides[index] if getattr(layout, "strides", None) is not None else None
    return Layout(
        shape=shape if isinstance(shape, tuple) else (shape,),
        strides=None if strides is None else (strides if isinstance(strides, tuple) else (strides,)),
    )


def take(layout: LayoutBase, begin: int, end: int) -> "Layout":
    """CuTe ``take<B, E>``: the modes in ``[begin, end)``, as a layout."""
    strides = getattr(layout, "strides", None)
    return Layout(
        shape=tuple(layout.shape[begin:end]),
        strides=None if strides is None else tuple(strides[begin:end]),
    )


__all__ = [
    "LayoutBase",
    "flatten",
    "size",
    "unflatten",
    "Layout",
    "Swizzle",
    "ComposedLayout",
    "EMPTY_LAYOUT",
    "get",
    "rank",
    "take",
]
