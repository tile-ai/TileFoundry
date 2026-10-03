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
    get,
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


def _coalesce_flat(layout: Layout, *, major: str = "col") -> Layout:
    """Apply the flat CuTe ``coalesce`` rule to one layout.

    Dimension interop imports core/types; defer until staged type imports
    finish. Likewise utils imports mesh, which imports layout_algebra.
    """
    from tilefoundry.ir.isl_interop import normalize_dim  # noqa: PLC0415

    from .utils import static_dim_value  # noqa: PLC0415

    if major not in {"col", "row"}:
        raise ValueError(f"coalesce: unknown major {major!r}")
    result_shape: list[int] = [1]
    result_stride: list[int] = [0]
    modes = tuple(zip(flat_shape(layout), flat_stride(layout)))
    for shape, stride in reversed(modes) if major == "row" else modes:
        shape, stride = normalize_dim(shape), normalize_dim(stride)
        if static_dim_value(shape) == 1:
            continue
        if result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        elif normalize_dim(result_shape[-1] * result_stride[-1]) == normalize_dim(stride):
            result_shape[-1] = normalize_dim(result_shape[-1] * shape)
        else:
            result_shape.append(shape)
            result_stride.append(stride)
    if major == "row":
        result_shape.reverse()
        result_stride.reverse()
    return Layout(shape=tuple(result_shape), strides=tuple(result_stride))


def _profile_place(path: tuple[int, ...]) -> str:
    return "profile" + "".join(f"[{index}]" for index in path)


def _coalesce_profile(
    layout: Layout, profile, path: tuple[int, ...], *, major: str, filtered: bool = False
) -> Layout:
    """Apply flat coalescing at the terminals selected by one profile."""
    if not isinstance(profile, tuple):
        return filter(layout, major=major) if filtered else _coalesce_flat(layout, major=major)

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
        child = _coalesce_profile(
            child, child_profile, (*path, index), major=major, filtered=filtered
        )
        if nested or isinstance(child_profile, tuple):
            result_shape.append(child.shape)
            result_stride.append(child.strides)
        else:
            result_shape.append(child.shape[0])
            result_stride.append(child.strides[0])
    return Layout(shape=tuple(result_shape), strides=tuple(result_stride))


def coalesce(
    layout: Union[Layout, ComposedLayout], trg_profile=_NO_PROFILE, *, major: str = "col"
):
    """CuTe ``coalesce``, optionally applied at ``trg_profile`` terminals.

    Coalescing renames the domain and leaves the index mapping alone, so a
    swizzled composed layout coalesces underneath its swizzle. A tuple profile
    transforms the corresponding top-level modes and retains any modes beyond
    its length, while a non-tuple terminal applies flat coalescing.
    """
    if get_swizzle_portion(layout) is not None:
        outer = (
            coalesce(layout.outer, major=major)
            if trg_profile is _NO_PROFILE
            else coalesce(layout.outer, trg_profile, major=major)
        )
        return ComposedLayout(inner=layout.inner, offset=layout.offset, outer=outer)
    if trg_profile is _NO_PROFILE:
        return _coalesce_flat(layout, major=major)
    return _coalesce_profile(layout, trg_profile, (), major=major)


def filter(layout: Union[Layout, ComposedLayout], profile=_NO_PROFILE, *, major: str = "col"):
    """CuTe ``filter``: drop shape-1 and stride-0 modes, then coalesce.

    Defer utils: it imports mesh, which imports layout_algebra at module load.
    """
    from .utils import static_dim_value  # noqa: PLC0415

    if get_swizzle_portion(layout) is not None:
        return ComposedLayout(
            inner=layout.inner, offset=layout.offset,
            outer=filter(layout.outer, profile, major=major),
        )
    if profile is not _NO_PROFILE:
        return _coalesce_profile(layout, profile, (), major=major, filtered=True)
    modes = tuple(
        (shape, stride) for shape, stride in zip(flat_shape(layout), flat_stride(layout))
        if static_dim_value(shape) != 1 and static_dim_value(stride) != 0
    )
    shape, strides = zip(*modes) if modes else ((1,), (0,))
    return coalesce(Layout(tuple(shape), tuple(strides)), major=major)


def is_contiguous(layout: Layout, *, major: str = "col") -> bool:
    """Whether an already filtered layout covers ``[0, size)`` without gaps.

    Pass the result of ``filter(layout, major=major)`` to avoid repeating that
    reduction. It has already coalesced in ``major`` order.
    Static extents use CuTe's ``size == cosize``. Symbolic extents use its
    structural equivalent: either coalescing direction leaves one mode
    stepping by one. Continuity does not depend on the order of modes.

    Defer utils: it imports mesh, which imports layout_algebra at module load.
    """
    from .utils import static_dim_value  # noqa: PLC0415

    if all(static_dim_value(shape) is not None for shape in flat_shape(layout)):
        return size(layout) == cosize(layout)
    if len(flat_shape(layout)) == 1 and flat_stride(layout) == (1,):
        return True
    other = coalesce(layout, major="row" if major == "col" else "col")
    return len(flat_shape(other)) == 1 and flat_stride(other) == (1,)


def complement(layout: Layout, max_idx: int = 1, *, major: str = "col") -> Layout:
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
    if major == "row":
        result_shape.reverse()
        result_stride.reverse()
    return coalesce(
        Layout(shape=tuple(result_shape), strides=tuple(result_stride)), major=major
    )


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


def _make_layout(*layouts: Layout) -> Layout:
    """CuTe ``make_layout``: collect each layout as one mode.

    Layout stores even scalar modes in a tuple; unwrap that representation
    when a mode has one leaf, retaining all nontrivial nesting.
    """
    def mode(values):
        return values[0] if len(values) == 1 else values

    return Layout(
        shape=tuple(mode(layout.shape) for layout in layouts),
        strides=tuple(
            mode(layout.strides if layout.strides is not None else compact_col_major(layout.shape))
            for layout in layouts
        ),
    )


def _compose_mode(left: Layout, shape: int, stride: int, *, major: str) -> Layout:
    """Compose one scalar mode using CuTe's strided-domain decomposition."""
    if stride == 0:
        return Layout((shape,), (0,))
    flat = coalesce(left, major=major)
    modes = tuple(zip(flat_shape(flat), flat_stride(flat)))
    if major == "row":
        modes = tuple(reversed(modes))
    result_shape, result_stride = [], []
    rest_shape, rest_stride = shape, stride
    for current_shape, current_stride in modes[:-1]:
        if not (current_shape % rest_stride == 0 or rest_stride % current_shape == 0):
            raise AssertionError
        new_shape = min(max(1, current_shape // rest_stride), rest_shape)
        if new_shape != 1:
            result_shape.append(new_shape)
            result_stride.append(rest_stride * current_stride)
        rest_shape //= new_shape
        rest_stride = -(-rest_stride // current_shape)
    if rest_shape != 1 or not result_shape:
        result_shape.append(rest_shape)
        result_stride.append(rest_stride * modes[-1][1])
    if major == "row":
        result_shape.reverse()
        result_stride.reverse()
    return Layout(tuple(result_shape), tuple(result_stride))


def _compose_layout(left: Layout, right, *, major: str) -> Layout:
    """CuTe's None, integer, tuple-of-tiles and Layout composition overloads."""
    if right is None:
        return left
    if isinstance(right, int):
        return _compose_mode(left, right, 1, major=major)
    if isinstance(right, tuple):
        if len(left.shape) < len(right):
            raise AssertionError
        return _make_layout(
            *(_compose_layout(get(left, index), tile, major=major) for index, tile in enumerate(right)),
            *(get(left, index) for index in range(len(right), len(left.shape))),
        )
    if isinstance(right, Layout):
        strides = right.strides if right.strides is not None else compact_col_major(right.shape)
        if len(right.shape) == 1 and not isinstance(right.shape[0], tuple):
            return _compose_mode(left, right.shape[0], strides[0], major=major)
        return _make_layout(
            *(_compose_layout(left, get(right, index), major=major) for index in range(len(right.shape)))
        )
    raise NotImplementedError(
        f"composition: no rule for {type(left).__name__} ∘ {type(right).__name__}"
    )


def composition(left, right, offset: int = 0, *, major: str = "col"):
    """CuTe ``composition``, dispatched to the supported overloads.

    The general Layout overload has no consumers in this round. Future
    consumers include ``utils._inner_layout`` and ``utils.tile_view_layout``;
    their currently pre-grouped tiles require a separate refactor. ``offset``
    belongs to the existing swizzle overload.
    """
    if swizzle_layout.supports_composition(left, right):
        return swizzle_layout.composition(left, right, offset)
    if isinstance(left, Layout):
        return _compose_layout(left, right, major=major)
    raise NotImplementedError(
        f"composition: no rule for {type(left).__name__} ∘ {type(right).__name__}"
    )


def logical_divide(layout: Layout, tile, *, major: str = "col") -> Layout:
    """CuTe ``logical_divide``: compose the integer tile and its complement.

    No consumers are connected in this round. ``utils._inner_layout`` and
    ``utils.tile_view_layout`` are future consumers after their pre-grouped
    tile representation is refactored. Symbolic tiles are unsupported.
    """
    if tile is None:
        return layout
    if isinstance(tile, tuple):
        if len(layout.shape) < len(tile):
            raise AssertionError
        return _make_layout(
            *(logical_divide(get(layout, index), one, major=major) for index, one in enumerate(tile)),
            *(get(layout, index) for index in range(len(tile), len(layout.shape))),
        )
    if isinstance(tile, int) and not isinstance(tile, bool):
        tile = Layout((tile,), (1,))
    if not isinstance(tile, Layout) or any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in (*flat_shape(tile), *flat_stride(tile))
    ):
        raise TypeError("logical_divide: tile must have integer extents and strides")
    return composition(
        layout, _make_layout(tile, complement(tile, size(layout), major=major)), major=major
    )


def zipped_divide(layout: Layout, tile, *, major: str = "col") -> Layout:
    """CuTe ``zipped_divide``: gather the divided tile and remainder modes.

    No consumers are connected in this round. ``utils._inner_layout`` and
    ``utils.tile_view_layout`` are future consumers after their pre-grouped
    tile representation is refactored. Symbolic tiles are unsupported.
    """
    if tile is None:
        return _make_layout(Layout((1,), (0,)), layout)
    if isinstance(tile, tuple):
        if len(layout.shape) < len(tile):
            raise AssertionError
        split = tuple(
            zipped_divide(get(layout, index), one, major=major) for index, one in enumerate(tile)
        )
        return _make_layout(
            _make_layout(*(get(one, 0) for one in split)),
            _make_layout(
                *(get(one, 1) for one in split),
                *(get(layout, index) for index in range(len(tile), len(layout.shape))),
            ),
        )
    return logical_divide(layout, tile, major=major)


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
    "filter",
    "is_contiguous",
    "complement",
    "is_inverse_projectable",
    "right_inverse",
    "left_inverse",
    "logical_divide",
    "zipped_divide",
]
