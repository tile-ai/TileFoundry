"""Provide the supported CuTe algebra for ``Layout`` and ``ComposedLayout``.

The restricted port covers coalescing, composition, complements, and inverses,
including the CuTe ``swizzle_layout.hpp`` specializations for a ``Swizzle`` in
``inner``. Layout application lives with the layout types themselves.

See [shard §9](docs/spec/shard.md#9-layout-construction-and-algebra).
"""

from __future__ import annotations

from typing import Union

from . import swizzle_layout
from .layout import (
    ComposedLayout,
    Layout,
    NotProjectable,
    Swizzle,
    apply,
    flat_shape,
    flat_stride,
    size,
)
from .stride import compact_col_major
from .swizzle_layout import get_swizzle_portion


def cosize(layout: Union[Layout, ComposedLayout]) -> int:
    """The codomain extent.

    A swizzle permutes bits inside the codomain its ``outer`` already spans,
    so it adds nothing to it: CuTe ``cosize`` of a swizzled composed layout is
    ``cosize`` of the layout underneath (``swizzle_layout.hpp:172``).
    """
    if get_swizzle_portion(layout) is not None:
        return cosize(layout.outer)
    return apply(layout, size(layout) - 1) + 1


_NO_PROFILE = object()


def _coalesce_flat(layout: Layout) -> Layout:
    """Apply the flat CuTe ``coalesce`` rule to one layout."""
    result_shape: list[int] = [1]
    result_stride: list[int] = [0]
    for shape, stride in zip(flat_shape(layout), flat_stride(layout)):
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


def _profile_place(path: tuple[int, ...]) -> str:
    return "profile" + "".join(f"[{index}]" for index in path)


def _coalesce_profile(layout: Layout, profile, path: tuple[int, ...]) -> Layout:
    """Apply flat coalescing at the terminals selected by one profile."""
    if not isinstance(profile, tuple):
        return _coalesce_flat(layout)

    strides = layout.strides
    if strides is None:
        strides = compact_col_major(layout.shape)
    if not isinstance(strides, tuple) or len(layout.shape) != len(strides):
        raise ValueError(
            f"coalesce: layout shape has {len(layout.shape)} modes at "
            f"{_profile_place(path)}, but its strides do not"
        )
    if len(profile) > len(layout.shape):
        raise ValueError(
            f"coalesce: {_profile_place(path)} has {len(profile)} modes, but the "
            f"layout there has {len(layout.shape)}"
        )

    result_shape: list = []
    result_stride: list = []
    for index, (shape, stride) in enumerate(zip(layout.shape, strides)):
        if index >= len(profile):
            result_shape.append(shape)
            result_stride.append(stride)
            continue

        nested = isinstance(shape, tuple)
        child = Layout(
            shape=shape if nested else (shape,),
            strides=stride if isinstance(stride, tuple) else (stride,),
        )
        child_profile = profile[index]
        child = _coalesce_profile(child, child_profile, (*path, index))
        if nested or isinstance(child_profile, tuple):
            result_shape.append(child.shape)
            result_stride.append(child.strides)
        else:
            result_shape.append(child.shape[0])
            result_stride.append(child.strides[0])
    return Layout(shape=tuple(result_shape), strides=tuple(result_stride))


def coalesce(layout: Union[Layout, ComposedLayout], trg_profile=_NO_PROFILE):
    """CuTe ``coalesce``, optionally applied at ``trg_profile`` terminals.

    Coalescing renames the domain and leaves the index mapping alone, so a
    swizzled composed layout coalesces underneath its swizzle. A tuple profile
    transforms the corresponding top-level modes and retains any modes beyond
    its length, while a non-tuple terminal applies flat coalescing.
    """
    if get_swizzle_portion(layout) is not None:
        outer = (
            coalesce(layout.outer)
            if trg_profile is _NO_PROFILE
            else coalesce(layout.outer, trg_profile)
        )
        return ComposedLayout(inner=layout.inner, offset=layout.offset, outer=outer)
    if trg_profile is _NO_PROFILE:
        return _coalesce_flat(layout)
    return _coalesce_profile(layout, trg_profile, ())


def complement(layout: Layout, max_idx: int = 1) -> Layout:
    """CuTe ``complement``: the modes that fill the gaps below ``max_idx``."""
    result_shape: list[int] = []
    result_stride: list[int] = []
    current_idx = 1
    for stride, shape in sorted(zip(flat_stride(layout), flat_shape(layout))):
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
    return Layout(shape=flat_shape(a) + flat_shape(b), strides=flat_stride(a) + flat_stride(b))


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
        (stride, shape)
        for shape, stride in zip(flat_shape(layout), flat_stride(layout))
        if shape != 1
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
    shape = flat_shape(layout)
    stride = flat_stride(layout)
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
        return flat_stride(inner) == compact_col_major(flat_shape(inner))
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


def composition(left, right, offset: int = 0):
    """CuTe ``composition``, dispatched to the supported overloads."""
    if swizzle_layout.supports_composition(left, right):
        return swizzle_layout.composition(left, right, offset)
    raise NotImplementedError(
        f"composition: no rule for {type(left).__name__} ∘ {type(right).__name__}"
    )


def left_inverse(layout: Union[Layout, ComposedLayout, Swizzle]):
    """CuTe ``left_inverse``, dispatched.

    Plain ``Layout`` → the flat algebra. ``ComposedLayout`` → the recursive
    rule ``inverse-of-composed = composition of component inverses``
    (``layout_composed.hpp:428``). For the v1-admissible identity-``inner`` case
    this is ``ComposedLayout(inner=left_inverse(outer), offset=-offset,
    outer=None)`` (``outer=None`` ≡ identity), i.e. ``image⁻¹(t) =
    outer⁻¹(t − offset)``.
    """
    if swizzle_layout.supports_inverse(layout):
        return swizzle_layout.inverse(layout, left_inverse)
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
    if swizzle_layout.supports_inverse(layout):
        return swizzle_layout.inverse(layout, right_inverse)
    if isinstance(layout, ComposedLayout):
        _check_admissible(layout)
        return ComposedLayout(
            inner=_right_inverse_layout(layout.outer), offset=-layout.offset, outer=None
        )
    if not is_inverse_projectable(layout):
        raise NotProjectable(f"{layout} is not inverse-projectable; no right inverse")
    return _right_inverse_layout(layout)


__all__ = [
    "composition",
    "cosize",
    "coalesce",
    "complement",
    "is_inverse_projectable",
    "right_inverse",
    "left_inverse",
]
