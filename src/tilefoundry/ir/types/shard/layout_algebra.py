"""Flat CuTe layout algebra for mesh execution scopes.

Ported from the pure-Python CuTe reference
(``third_party/cutlass/python/pycute/{int_tuple,layout}.py``), restricted to
the **flat** (non-hierarchical) ``Layout`` / ``ComposedLayout`` the shard IR
uses. The ported pieces are exactly what the mesh model needs:

- ``apply`` — ``crd2idx`` of a 1-D domain coord (the layout as a function);
- ``left_inverse`` / ``right_inverse`` — the CuTe inverses, used to recover a
  coordinate from a thread index;
- ``contains`` / ``project`` over a ``ComposedLayout`` execution scope:
  ``project`` is ``left_inverse`` applied to the thread index, ``contains`` adds
  the domain-bounds + round-trip check (see ``docs/plans/mesh-warp-specialization.md``).

``image(c) = inner(offset + outer(c))`` matches CuTeDSL ``make_composed_layout``.
Only inverse-projectable (injective, compact-image) layouts are admissible
execution scopes; a non-injective layout raises ``NotProjectable``.
"""
from __future__ import annotations

from math import prod
from typing import Optional, Union

from .layout import ComposedLayout, Layout


class NotProjectable(ValueError):
    """A layout cannot serve as a mesh execution scope (not inverse-projectable)."""


def _shape(layout: Layout) -> tuple[int, ...]:
    s = layout.shape
    return s if isinstance(s, tuple) else (s,)


def _stride(layout: Layout) -> tuple[int, ...]:
    if layout.strides is not None:
        return layout.strides
    return prefix_product(_shape(layout))


def prefix_product(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Exclusive prefix product (column-major natural strides)."""
    out: list[int] = []
    acc = 1
    for s in shape:
        out.append(acc)
        acc *= s
    return tuple(out)


def size(layout: Layout) -> int:
    return prod(_shape(layout))


def apply(layout: Layout, coord: int) -> int:
    """``crd2idx`` of a 1-D domain coord: decompose by shape, dot with strides."""
    shape = _shape(layout)
    stride = _stride(layout)
    idx = 0
    rem = coord
    for s, d in zip(shape, stride):
        idx += (rem % s) * d
        rem //= s
    return idx


def cosize(layout: Layout) -> int:
    return apply(layout, size(layout) - 1) + 1


def idx2crd(idx: int, shape: tuple[int, ...], stride: tuple[int, ...]) -> tuple[int, ...]:
    """Per-mode ``(idx // stride_i) % shape_i`` (CuTe ``idx2crd``)."""
    return tuple((idx // d) % s for s, d in zip(shape, stride))


def coalesce(layout: Layout) -> Layout:
    """Flatten + merge contiguous modes, drop shape-1 modes (CuTe ``coalesce``)."""
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
    result_shape.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
    result_stride.append(current_idx)
    return coalesce(Layout(shape=tuple(result_shape), strides=tuple(result_stride)))


def _make_flat(a: Layout, b: Layout) -> Layout:
    """Concatenate two flat layouts into one (CuTe ``make_layout`` after flatten)."""
    return Layout(shape=_shape(a) + _shape(b), strides=_stride(a) + _stride(b))


def is_inverse_projectable(layout: Layout) -> bool:
    """Can ``layout`` be inverted by the CuTe ``left_inverse`` algorithm (and so
    serve as a mesh execution scope) — i.e. is it injective *and* compact-ordered.

    Drop shape-1 modes, sort the rest by stride; each stride must be non-zero
    (no broadcast collision) and a multiple of the running codomain extent
    ``Π prev (stride*shape)`` (so the gap is integral and modes do not overlap).
    Necessary and sufficient for ``left_inverse(layout)`` to round-trip.
    Note ``(5,3):(3,8)`` is injective yet NOT projectable (``8 % 15 != 0``)."""
    current = 1
    modes = sorted(
        (stride, shape)
        for shape, stride in zip(_shape(layout), _stride(layout))
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
    shape = _shape(layout)
    stride = _stride(layout)
    triples = sorted(zip(stride, shape, prefix_product(shape)))
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


# --- ComposedLayout: identity-inner admissibility + recursive inverse --------


def _is_identity_inner(inner: object) -> bool:
    """An ``inner`` that is ``None`` or a unit-stride contiguous layout acts as
    identity on the ``offset + outer(coord)`` index (the mesh affine case)."""
    if inner is None:
        return True
    if isinstance(inner, Layout):
        return _stride(inner) == prefix_product(_shape(inner))
    return False


def _check_admissible(scope: ComposedLayout) -> None:
    """A ``ComposedLayout`` is an admissible mesh execution scope only if its
    ``inner`` is identity (v1) and its ``outer`` is a plain injective ``Layout``.
    Anything else fails closed with ``NotProjectable`` (no colliding fallback)."""
    if not _is_identity_inner(scope.inner):
        raise NotProjectable("non-identity inner is not an admissible mesh scope (v1)")
    if not isinstance(scope.outer, Layout):
        raise NotProjectable("outer must be a plain Layout for a mesh scope")
    if not is_inverse_projectable(scope.outer):
        raise NotProjectable("outer layout is not inverse-projectable (injective + compact)")


def left_inverse(layout: Union[Layout, ComposedLayout]):
    """CuTe ``left_inverse``, dispatched.

    Plain ``Layout`` → the flat algebra. ``ComposedLayout`` → the recursive
    rule ``inverse-of-composed = composition of component inverses``
    (``layout_composed.hpp:428``). For the v1-admissible identity-``inner`` case
    this is ``ComposedLayout(inner=left_inverse(outer), offset=-offset,
    outer=None)`` (``outer=None`` ≡ identity), i.e. ``image⁻¹(t) =
    outer⁻¹(t − offset)``."""
    if isinstance(layout, ComposedLayout):
        _check_admissible(layout)
        return ComposedLayout(
            inner=_left_inverse_layout(layout.outer), offset=-layout.offset, outer=None
        )
    if not is_inverse_projectable(layout):
        raise NotProjectable(f"{layout} is not inverse-projectable; no left inverse")
    return _left_inverse_layout(layout)


def right_inverse(layout: Union[Layout, ComposedLayout]):
    """CuTe ``right_inverse``, dispatched (mirror of :func:`left_inverse`)."""
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
    """Recover the multi-dim domain coord ``(warp, lane, …)`` of thread ``t``,
    or ``None`` if ``t`` is not in this scope. Raises ``NotProjectable`` if the
    scope itself is inadmissible (non-identity inner / non-injective outer).

    Built on the composed ``left_inverse``: ``coord_1d = left_inverse(scope)(t)``
    (``= left_inverse(outer)(t − offset)``); the multi-dim coord is ``idx2crd``
    of that over ``outer``'s shape. Returns ``None`` unless the coord is
    in-domain *and* round-trips (``image(coord) == t``)."""
    _check_admissible(scope)
    outer = scope.outer
    if t - scope.offset < 0:
        return None
    coord_1d = _apply_any(left_inverse(scope), t)
    if not (0 <= coord_1d < size(outer)):
        return None
    # round-trip: reject thread indices not in outer's image (stride remainder)
    if image(scope, coord_1d) != t:
        return None
    # coord_1d is the 1-D domain linearization; split it over the domain's
    # natural (prefix-product) strides, NOT outer's image strides.
    shape = _shape(outer)
    return idx2crd(coord_1d, shape, prefix_product(shape))


def contains(scope: ComposedLayout, t: int) -> bool:
    """Does thread ``t`` execute this mesh scope's body."""
    return project(scope, t) is not None


__all__ = [
    "NotProjectable",
    "prefix_product",
    "size",
    "cosize",
    "apply",
    "idx2crd",
    "coalesce",
    "complement",
    "is_inverse_projectable",
    "right_inverse",
    "left_inverse",
    "image",
    "project",
    "contains",
]
