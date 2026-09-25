"""Provide flat CuTe layout algebra for mesh execution scopes.

The restricted port supports coordinate application, inverses, containment,
and projection for ``Layout`` and ``ComposedLayout``, and the CuTe
``swizzle_layout.hpp`` specializations for a ``Swizzle`` in ``inner``.
Execution scopes must be injective and inverse-projectable.

See [shard §9](docs/spec/shard.md#9-layout-construction-and-mesh-scope-projection).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Union

from tilefoundry.ir.types.layout import flatten

from .layout import ComposedLayout, Layout, Swizzle, size
from .shard_layout import ShardLayout
from .stride import compact_col_major, idx2crd


class NotProjectable(ValueError):
    """A layout cannot serve as a mesh execution scope (not inverse-projectable)."""


ASYNC_WIDTHS = (4, 8, 16)


@dataclass(frozen=True)
class Run:
    """One contiguous run of modes from one logical tile axis."""

    extent: int
    step: int
    axis: int
    mode: int


def _shape(layout: Layout) -> tuple[int, ...]:
    return flatten(layout.shape)


def _stride(layout: Layout) -> tuple[int, ...]:
    if layout.strides is not None:
        return layout.strides
    return compact_col_major(_shape(layout))


def swizzle_of(layout: object) -> Optional[Swizzle]:
    """The ``Swizzle`` a composed layout applies last, or ``None``.

    CuTe ``get_swizzle_portion``, answering ``None`` rather than the identity
    ``Swizzle<0,4,3>`` so a caller can branch on "is this swizzled at all".
    """
    if isinstance(layout, ComposedLayout) and isinstance(layout.inner, Swizzle):
        return layout.inner
    return None


def apply(layout: Union[Layout, ComposedLayout], coord: int) -> int:
    """``crd2idx`` of a 1-D domain coord: decompose by shape, dot with strides.

    A ``ComposedLayout`` applies its components in order, so this is
    ``inner(offset + outer(coord))`` — the swizzle case included, since a
    ``Swizzle`` is exactly a mapping on that index.
    """
    if isinstance(layout, ComposedLayout):
        return _apply_any(layout, coord)
    shape = _shape(layout)
    stride = _stride(layout)
    idx = 0
    rem = coord
    for s, d in zip(shape, stride):
        idx += (rem % s) * d
        rem //= s
    return idx


def _crd2idx_unbounded(layout: Layout, coord: int) -> int:
    """``apply`` as CuTe's ``crd2idx`` has it: the last mode is not wrapped.

    ``apply`` wraps every mode, which is the same answer for an in-domain
    coord and the one this module's callers want. The swizzle composition
    rules feed a *bit mask* through a layout instead, which is routinely
    larger than the domain, and CuTe leaves the final mode unwrapped so those
    high bits keep contributing. Only that port reads this.
    """
    shape = _shape(layout)
    stride = _stride(layout)
    idx = 0
    rem = coord
    last = len(shape) - 1
    for position, (s, d) in enumerate(zip(shape, stride)):
        if position == last:
            idx += rem * d
        else:
            idx += (rem % s) * d
            rem //= s
    return idx


def cosize(layout: Union[Layout, ComposedLayout]) -> int:
    """The codomain extent.

    A swizzle permutes bits inside the codomain its ``outer`` already spans,
    so it adds nothing to it: CuTe ``cosize`` of a swizzled composed layout is
    ``cosize`` of the layout underneath (``swizzle_layout.hpp:172``).
    """
    if swizzle_of(layout) is not None:
        return cosize(layout.outer)
    return apply(layout, size(layout) - 1) + 1


def coalesce(layout: Union[Layout, ComposedLayout]):
    """Flatten + merge contiguous modes, drop shape-1 modes (CuTe ``coalesce``).

    Coalescing renames the domain and leaves the index mapping alone, so a
    swizzled composed layout coalesces underneath its swizzle.
    """
    if swizzle_of(layout) is not None:
        return ComposedLayout(
            inner=layout.inner, offset=layout.offset, outer=coalesce(layout.outer)
        )
    result_shape: list[int] = [1]
    result_stride: list[int] = [0]
    for shape, stride in zip(_shape(layout), _stride(layout)):
        if shape == 1:
            continue
        if result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        elif result_shape[-1] * result_stride[-1] == stride:
            result_shape[-1] = result_shape[-1] * shape
        else:
            result_shape.append(shape)
            result_stride.append(stride)
    return Layout(shape=tuple(result_shape), strides=tuple(result_stride))


def frame_of(layout: Union[Layout, ComposedLayout]) -> tuple[int, Layout] | None:
    """Read a bare layout or an affine composition as offset + outer layout."""
    if isinstance(layout, Layout):
        return 0, layout
    if layout.inner is not None or not isinstance(layout.outer, Layout):
        return None
    return layout.offset, layout.outer


def box_runs(
    layout: Layout,
    element_bits: int,
    span: int | None,
    limit: int | None = 256,
) -> tuple[Run, ...]:
    """Read contiguous runs by tile axis, ordered by increasing step."""
    runs: list[Run] = []
    for axis, (extents, steps) in enumerate(zip(layout.shape, layout.strides)):
        modes = tuple(enumerate(zip(flatten(extents), flatten(steps))))
        for mode, (extent, step) in reversed(modes):
            if extent == 1:
                continue
            last = runs[-1] if runs and runs[-1].axis == axis else None
            joined = None if last is None else last.extent * extent
            if (
                last is not None
                and step == last.step * last.extent
                and (limit is None or joined <= limit)
                and not (
                    span is not None
                    and last.step == 1
                    and joined * element_bits > span * 8
                )
            ):
                runs[-1] = replace(last, extent=joined)
            else:
                runs.append(Run(extent, step, axis, mode))
    return tuple(sorted(runs, key=lambda run: run.step))


def vector_widths(layout, element_bits: int) -> tuple[int, ...]:
    """Every cp.async width that divides every run in an arrangement."""
    from tilefoundry.ir.pattern.constraint import affine_part  # noqa: PLC0415

    widest = ASYNC_WIDTHS[-1]
    if isinstance(layout, ShardLayout):
        layout = layout.layout
    inner = getattr(layout, "inner", None)
    if inner is not None and hasattr(inner, "base"):
        widest = min(widest, 1 << inner.base)
    held = affine_part(layout)
    if held is None or any(
        type(value) is not int
        for group in (held.shape, held.strides)
        for value in flatten(group)
    ):
        return ()
    runs = box_runs(held, element_bits, None, limit=None)
    unit = [run.extent for run in runs if run.step == 1]
    if len(unit) != 1:
        return ()
    counted = (unit[0], *(run.step for run in runs if run.step != 1))
    return tuple(
        width
        for width in ASYNC_WIDTHS
        if width <= widest
        and all(value * element_bits % (width * 8) == 0 for value in counted)
    )


def complement(layout: Layout, max_idx: int = 1) -> Layout:
    """CuTe ``complement``: the modes that fill the gaps below ``max_idx``."""
    result_shape: list[int] = []
    result_stride: list[int] = []
    current_idx = 1
    for stride, shape in sorted(zip(_stride(layout), _shape(layout))):
        if stride == 0 or shape == 1:
            continue
        if current_idx > shape * stride:
            raise NotProjectable("complement: layout modes overlap (not invertible)")
        result_shape.append(stride // current_idx)
        result_stride.append(current_idx)
        current_idx = shape * stride
    result_shape.append((max_idx + current_idx - 1) // current_idx)
    result_stride.append(current_idx)
    return coalesce(Layout(shape=tuple(result_shape), strides=tuple(result_stride)))


def _make_flat(a: Layout, b: Layout) -> Layout:
    """Concatenate two flat layouts into one (CuTe ``make_layout`` after flatten)."""
    return Layout(shape=_shape(a) + _shape(b), strides=_stride(a) + _stride(b))


def is_inverse_projectable(layout: Layout) -> bool:
    """Can ``layout`` be inverted by the CuTe ``left_inverse`` algorithm — i.e.

    Can ``layout`` be inverted by the CuTe ``left_inverse`` algorithm (and so
    serve as a mesh execution scope) — i.e. is it injective *and* compact-ordered.

    Drop shape-1 modes, sort the rest by stride; each stride must be non-zero
    (no broadcast collision) and a multiple of the running codomain extent
    ``Π prev (stride*shape)`` (so the gap is integral and modes do not overlap).
    Necessary and sufficient for ``left_inverse(layout)`` to round-trip.
    Note ``(5,3):(3,8)`` is injective yet NOT projectable (``8 % 15 != 0``).
    """
    current = 1
    modes = sorted(
        (stride, shape) for shape, stride in zip(_shape(layout), _stride(layout)) if shape != 1
    )
    for stride, shape in modes:
        if stride == 0 or stride % current != 0:
            return False
        current = stride * shape
    return True


def _right_inverse_layout(layout: Layout) -> Layout:
    """CuTe ``right_inverse``: ``layout(right_inverse(layout)(i)) == i``."""
    result_shape: list[int] = []
    result_stride: list[int] = []
    current_idx = 1
    shape = _shape(layout)
    stride = _stride(layout)
    triples = sorted(zip(stride, shape, compact_col_major(shape)))
    for st, sh, rstride in triples:
        if sh == 1:
            continue
        if current_idx != st:
            break
        result_shape.append(sh)
        result_stride.append(rstride)
        current_idx = sh * st
    return coalesce(Layout(shape=tuple(result_shape), strides=tuple(result_stride)))


def _left_inverse_layout(layout: Layout) -> Layout:
    """CuTe ``left_inverse``: ``left_inverse(layout)(layout(i)) == i`` (injective)."""
    return _right_inverse_layout(_make_flat(layout, complement(layout)))


def _is_identity_inner(inner: object) -> bool:
    """Return whether identity inner.

    An ``inner`` that is ``None`` or a unit-stride contiguous layout acts as
    identity on the ``offset + outer(coord)`` index (the mesh affine case).
    """
    if inner is None:
        return True
    if isinstance(inner, Layout):
        return _stride(inner) == compact_col_major(_shape(inner))
    return False


def _check_admissible(scope: ComposedLayout) -> None:
    """Check admissible.

    A ``ComposedLayout`` is an admissible mesh execution scope only if its
    ``inner`` is identity (v1) and its ``outer`` is a plain injective ``Layout``.
    Anything else fails closed with ``NotProjectable`` (no colliding fallback).
    """
    if not _is_identity_inner(scope.inner):
        raise NotProjectable("non-identity inner is not an admissible mesh scope (v1)")
    if not isinstance(scope.outer, Layout):
        raise NotProjectable("outer must be a plain Layout for a mesh scope")
    if not is_inverse_projectable(scope.outer):
        raise NotProjectable("outer layout is not inverse-projectable (injective + compact)")


def _make_swizzle(active_y: int, active_z: int) -> Swizzle:
    """CuTe ``make_swizzle<Y,Z>()``: the swizzle that XORs *Y* onto *Z*.

    The two masks must hold the same number of bits; their trailing-zero
    counts give ``base`` and the signed ``shift``, and the reconstructed
    ``swizzle_code`` must give the masks back, which is how CuTe checks that
    the pair is a swizzle it can represent at all.
    """
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


def composition(left, right, offset: int = 0):
    """CuTe ``composition`` for the swizzle cases (``swizzle_layout.hpp:302``).

    ``composition(Swizzle, Layout)`` builds a swizzled layout, which states
    ``Swizzle(offset + Layout(coord))``.

    ``composition(Layout, Swizzle)`` would otherwise want the ``Swizzle`` in
    ``outer``, which has no domain to be a domain-side component of. CuTe
    instead reads which of the swizzle's bits the layout leaves active,
    rebuilds a ``Swizzle`` over those, and puts it back on the inner side.
    """
    if isinstance(left, Swizzle) and isinstance(right, Layout):
        if left.bits == 0 and offset == 0:
            return right
        return ComposedLayout(inner=left, offset=offset, outer=right)
    if isinstance(left, Layout) and isinstance(right, Swizzle):
        if offset:
            raise NotImplementedError(
                f"composition: a non-zero offset ({offset}) between a Layout and a "
                f"Swizzle has no canonical ComposedLayout form"
            )
        active_y = _crd2idx_unbounded(left, right.yyy_mask)
        active_z = _crd2idx_unbounded(left, right.zzz_mask)
        return composition(_make_swizzle(active_y, active_z), left)
    raise NotImplementedError(
        f"composition: no rule for {type(left).__name__} ∘ {type(right).__name__}"
    )


def _swizzled_inverse(layout: ComposedLayout, inverse_of_layout):
    """CuTe's swizzled ``left_inverse``/``right_inverse`` (``swizzle_layout.hpp:344``).

    ``inverse(Swizzle(offset + outer(c)))`` passes the swizzle back to the
    left of the inverted ``outer``, which ``composition(Layout, Swizzle)``
    then canonicalizes back into this IR's one legal shape. CuTe's non-zero
    ``offset`` branch composes ``inverse(offset)`` between the two, which
    lands a bare ``Swizzle`` in ``outer``; that is not a layout, so this
    refuses it by name rather than building something unrepresentable.
    """
    if layout.offset != 0:
        raise NotImplementedError(
            f"inverse: a swizzled composed layout with a non-zero offset "
            f"({layout.offset}) inverts to a Swizzle on the domain side, which "
            f"ComposedLayout.outer cannot hold"
        )
    if not isinstance(layout.outer, Layout):
        raise NotImplementedError(
            f"inverse: a swizzled composed layout inverts through its outer "
            f"Layout; this one states {type(layout.outer).__name__}"
        )
    return composition(inverse_of_layout(layout.outer), layout.inner)


def left_inverse(layout: Union[Layout, ComposedLayout, Swizzle]):
    """CuTe ``left_inverse``, dispatched.

    Plain ``Layout`` → the flat algebra. ``ComposedLayout`` → the recursive
    rule ``inverse-of-composed = composition of component inverses``
    (``layout_composed.hpp:428``). For the v1-admissible identity-``inner`` case
    this is ``ComposedLayout(inner=left_inverse(outer), offset=-offset,
    outer=None)`` (``outer=None`` ≡ identity), i.e. ``image⁻¹(t) =
    outer⁻¹(t − offset)``.
    """
    if isinstance(layout, Swizzle):
        return layout
    if swizzle_of(layout) is not None:
        return _swizzled_inverse(layout, left_inverse)
    if isinstance(layout, ComposedLayout):
        _check_admissible(layout)
        return ComposedLayout(
            inner=_left_inverse_layout(layout.outer), offset=-layout.offset, outer=None
        )
    if not is_inverse_projectable(layout):
        raise NotProjectable(f"{layout} is not inverse-projectable; no left inverse")
    return _left_inverse_layout(layout)


def right_inverse(layout: Union[Layout, ComposedLayout, Swizzle]):
    """CuTe ``right_inverse``, dispatched (mirror of :func:`left_inverse`).

    A ``Swizzle`` is an involution -- its Y and Z bit ranges do not overlap --
    so it is its own inverse on both sides (``swizzle_layout.hpp:371``).
    """
    if isinstance(layout, Swizzle):
        return layout
    if swizzle_of(layout) is not None:
        return _swizzled_inverse(layout, right_inverse)
    if isinstance(layout, ComposedLayout):
        _check_admissible(layout)
        return ComposedLayout(
            inner=_right_inverse_layout(layout.outer), offset=-layout.offset, outer=None
        )
    if not is_inverse_projectable(layout):
        raise NotProjectable(f"{layout} is not inverse-projectable; no right inverse")
    return _right_inverse_layout(layout)


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


def image(scope: ComposedLayout, coord: int) -> int:
    """``inner(offset + outer(coord))`` for a 1-D domain coord."""
    _check_admissible(scope)
    return scope.offset + apply(scope.outer, coord)


def project(scope: ComposedLayout, t: int) -> Optional[tuple[int, ...]]:
    """Project.

    Recover the multi-dim domain coord ``(warp, lane, …)`` of thread ``t``,
    or ``None`` if ``t`` is not in this scope. Raises ``NotProjectable`` if the
    scope itself is inadmissible (non-identity inner / non-injective outer).

    Built on the composed ``left_inverse``: ``coord_1d = left_inverse(scope)(t)``
    (``= left_inverse(outer)(t − offset)``); the multi-dim coord is ``idx2crd``
    of that over ``outer``'s shape. Returns ``None`` unless the coord is
    in-domain *and* round-trips (``image(coord) == t``).
    """
    _check_admissible(scope)
    outer = scope.outer
    if t - scope.offset < 0:
        return None
    coord_1d = _apply_any(left_inverse(scope), t)
    if not (0 <= coord_1d < size(outer)):
        return None

    if image(scope, coord_1d) != t:
        return None

    shape = _shape(outer)
    return idx2crd(coord_1d, shape, compact_col_major(shape))


def contains(scope: ComposedLayout, t: int) -> bool:
    """Does thread ``t`` execute this mesh scope's body."""
    return project(scope, t) is not None


__all__ = [
    "ASYNC_WIDTHS",
    "NotProjectable",
    "Run",
    "swizzle_of",
    "composition",
    "cosize",
    "apply",
    "coalesce",
    "frame_of",
    "box_runs",
    "complement",
    "is_inverse_projectable",
    "right_inverse",
    "left_inverse",
    "image",
    "project",
    "contains",
    "vector_widths",
]
