"""Layouts: shape and strides of an arrangement, the swizzle functor, and their application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tilefoundry import services
from tilefoundry.ir import expr

from . import int_tuple
from .int_tuple import product
from .stride import compact_row_major

__all__ = [
    "ComposedLayout",
    "Layout",
    "LayoutBase",
    "NotProjectableError",
    "Swizzle",
    "apply",
    "flat_shape",
    "flat_stride",
    "flatten",
    "get",
    "normalize_strides",
    "rank",
    "size",
    "take",
    "unflatten",
]


class LayoutBase:
    """Common domain-shape contract for tensor layout descriptors."""

    @property
    def domain_rank(self) -> int:
        """How many modes the domain shape has once flattened."""
        return len(flatten(self.shape))


class NotProjectableError(ValueError):
    """A layout cannot serve as a mesh execution scope (not inverse-projectable)."""


@dataclass(frozen=True)
class Layout(LayoutBase):
    """Cute-style layout: shape + per-axis cute strides."""

    shape: tuple["ShapeDim", ...]
    strides: Optional[tuple["ShapeDim", ...]] = None

    def __post_init__(self):
        """Fill in compact row-major strides when none are stated."""
        if self.strides is not None:
            return
        shape = canonical_mode(tuple(self.shape))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self, "strides", normalize_strides(shape, compact_row_major(shape, mul=dim_multiply))
        )


def canonical_mode(mode):
    """*mode* with every integer ``Constant`` replaced by its plain integer."""
    if isinstance(mode, tuple):
        return tuple(canonical_mode(child) for child in mode)
    if isinstance(mode, int):
        return mode
    return mode.value if isinstance(mode, expr.Constant) and isinstance(mode.value, int) else mode


def dim_multiply(a, b):
    """The product of two extents: an integer when both are, else a dimension expression."""
    if isinstance(a, int) and isinstance(b, int):
        return a * b
    return services.simplify_dim("mul", (a, b))


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

    @classmethod
    def from_active(cls, active_y: int, active_z: int) -> Swizzle:
        """CuTe ``make_swizzle<Y,Z>()``: build the swizzle XORing *Y* onto *Z*."""
        bits_y, bits_z = active_y.bit_count(), active_z.bit_count()
        if bits_y != bits_z:
            raise NotImplementedError(
                f"composition: the Y mask {active_y:#x} holds {bits_y} bits and the Z "
                f"mask {active_z:#x} holds {bits_z}; only an equal-width pair is a "
                f"Swizzle"
            )
        if bits_y == 0:
            return cls(0, 0, 0)
        trailing_y = (active_y & -active_y).bit_length() - 1
        trailing_z = (active_z & -active_z).bit_length() - 1
        swizzle = cls(bits_y, min(trailing_y, trailing_z), trailing_y - trailing_z)
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
        """The domain shape: the outer layout's, else the inner one's."""
        domain = self.outer if self.outer is not None else self.inner
        if domain is None or isinstance(domain, Swizzle):
            return ()
        return domain.shape


def flat_shape(layout: Layout) -> tuple[int, ...]:
    """Return CuTe ``flatten(layout.shape())`` as a flat tuple."""
    return int_tuple.flatten(layout.shape)


def flat_stride(layout: Layout) -> tuple[int, ...]:
    """Flatten a concrete layout's strides."""
    return int_tuple.flatten(layout.strides)


def apply(layout: Layout | ComposedLayout, coord: int) -> int:
    """``crd2idx`` of a 1-D domain coord: decompose by shape, dot with strides.

    A ``ComposedLayout`` applies its components in order, so this is
    ``inner(offset + outer(coord))`` — the swizzle case included, since a
    ``Swizzle`` is exactly a mapping on that index.
    """
    if isinstance(layout, ComposedLayout):
        return apply_any(layout, coord)
    shape = flat_shape(layout)
    stride = flat_stride(layout)
    idx = 0
    rem = coord
    for extent, step in zip(shape, stride):
        idx += (rem % extent) * step
        rem //= extent
    return idx


def apply_any(layout, x: int) -> int:
    """Apply a ``Layout`` / ``ComposedLayout`` (``None`` ≡ identity) to ``x``."""
    if layout is None:
        return x
    if isinstance(layout, Swizzle):
        return layout(x)
    if isinstance(layout, Layout):
        return apply(layout, x)
    if isinstance(layout, ComposedLayout):
        return apply_any(layout.inner, layout.offset + apply_any(layout.outer, x))
    raise NotProjectableError(f"cannot apply layout of type {type(layout).__name__}")


def size(layout: Layout) -> int:
    """The number of positions in the layout's domain."""
    return product(layout.shape)


def flatten(layout):
    """CuTe ``flatten``: every mode at the top level, of an arrangement or a tuple.

    CuTe spells this once for each (``layout.hpp`` and the tuple algorithms);
    here one name reads both, because which was handed over is plain from what
    comes back.
    """
    if not isinstance(layout, LayoutBase):
        return int_tuple.flatten(layout)
    if isinstance(layout, ComposedLayout):
        return flatten(layout.outer) if layout.outer is not None else Layout((), ())
    strides = layout.strides
    return Layout(
        shape=int_tuple.flatten(layout.shape),
        strides=int_tuple.flatten(strides),
    )


def unflatten(layout: LayoutBase, profile) -> "Layout":
    """CuTe ``unflatten``: a flat arrangement nested to *profile*'s shape."""
    strides = layout.strides
    return Layout(
        shape=int_tuple.unflatten(tuple(layout.shape), profile),
        strides=int_tuple.unflatten(tuple(strides), profile),
    )


def rank(layout: LayoutBase) -> int:
    """CuTe ``rank``: how many modes a layout states at its top level."""
    return len(layout.shape)


def get(layout: LayoutBase, index: int) -> "Layout":
    """CuTe ``get<I>``: one mode of a layout, as a layout of its own."""
    shape = layout.shape[index]
    strides = layout.strides[index]
    return Layout(
        shape=shape if isinstance(shape, tuple) else (shape,),
        strides=strides if isinstance(strides, tuple) else (strides,),
    )


def take(layout: LayoutBase, begin: int, end: int) -> "Layout":
    """CuTe ``take<B, E>``: the modes in ``[begin, end)``, as a layout."""
    strides = layout.strides
    return Layout(
        shape=tuple(layout.shape[begin:end]),
        strides=tuple(strides[begin:end]),
    )


def normalize_strides(shape, strides):
    """*strides* with a zero step wherever the matching extent is one."""
    if isinstance(shape, tuple):
        return tuple(
            normalize_strides(one, step) for one, step in zip(shape, strides, strict=True)
        )
    extent = shape.value if isinstance(shape, expr.Constant) else shape
    return (
        0 if isinstance(extent, int) and not isinstance(extent, bool) and extent == 1 else strides
    )
