from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .int_tuple import flatten as _flat
from .int_tuple import product
from .stride import compact_col_major


class LayoutBase:
    """Common domain-shape contract for tensor layout descriptors."""

    @property
    def domain_rank(self) -> int:
        return len(flatten(self.shape))


class NotProjectable(ValueError):
    """A layout cannot serve as a mesh execution scope (not inverse-projectable)."""


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


def make_swizzle(active_y: int, active_z: int) -> Swizzle:
    """CuTe ``make_swizzle<Y,Z>()``: build the swizzle XORing *Y* onto *Z*."""
    bits_y, bits_z = active_y.bit_count(), active_z.bit_count()
    if bits_y != bits_z:
        raise NotImplementedError(
            f"composition: the Y mask {active_y:#x} holds {bits_y} bits and the Z "
            f"mask {active_z:#x} holds {bits_z}; only an equal-width pair is a "
            f"Swizzle"
        )
    if bits_y == 0:
        return Swizzle(0, 0, 0)
    trailing_y = (active_y & -active_y).bit_length() - 1
    trailing_z = (active_z & -active_z).bit_length() - 1
    swizzle = Swizzle(bits_y, min(trailing_y, trailing_z), trailing_y - trailing_z)
    if swizzle.swizzle_code != (active_y | active_z):
        raise NotImplementedError(
            f"composition: the mask pair ({active_y:#x}, {active_z:#x}) is not a "
            f"Swizzle<B,M,S>; its bits are not two contiguous equal-width runs"
        )
    return swizzle


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


def flat_shape(layout: Layout) -> tuple[int, ...]:
    """Return CuTe ``flatten(layout.shape())`` as a flat tuple."""
    return _flat(layout.shape)


def flat_stride(layout: Layout) -> tuple[int, ...]:
    """Flatten stated strides, or synthesize the compact column-major default."""
    if layout.strides is not None:
        return _flat(layout.strides)
    return compact_col_major(flat_shape(layout))


def apply(layout: Layout | ComposedLayout, coord: int) -> int:
    """``crd2idx`` of a 1-D domain coord: decompose by shape, dot with strides.

    A ``ComposedLayout`` applies its components in order, so this is
    ``inner(offset + outer(coord))`` — the swizzle case included, since a
    ``Swizzle`` is exactly a mapping on that index.
    """
    if isinstance(layout, ComposedLayout):
        return _apply_any(layout, coord)
    shape = flat_shape(layout)
    stride = flat_stride(layout)
    idx = 0
    rem = coord
    for extent, step in zip(shape, stride):
        idx += (rem % extent) * step
        rem //= extent
    return idx


def _apply_any(layout, x: int) -> int:
    """Apply a ``Layout`` / ``ComposedLayout`` (``None`` ≡ identity) to ``x``."""
    if layout is None:
        return x
    if isinstance(layout, Swizzle):
        return layout(x)
    if isinstance(layout, Layout):
        return apply(layout, x)
    if isinstance(layout, ComposedLayout):
        return _apply_any(layout.inner, layout.offset + _apply_any(layout.outer, x))
    raise NotProjectable(f"cannot apply layout of type {type(layout).__name__}")


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
    "NotProjectable",
    "apply",
    "flatten",
    "size",
    "unflatten",
    "Layout",
    "Swizzle",
    "ComposedLayout",
    "EMPTY_LAYOUT",
    "flat_shape",
    "flat_stride",
    "get",
    "make_swizzle",
    "rank",
    "take",
]
