"""IntTuple alias + helpers."""

from __future__ import annotations

from typing import Union, overload

from tilefoundry.ir.types.tensor_type import ShapeDim

IntTuple = Union[int, tuple["IntTuple", ...]]


@overload
def flatten(t: IntTuple) -> tuple[int, ...]: ...


@overload
def flatten(t: object) -> tuple[object, ...]: ...


def flatten(t: object) -> tuple[object, ...]:
    if not isinstance(t, tuple):
        return (t,)
    return tuple(value for item in t for value in flatten(item))


def partition_modes(modes, count, indivisible=()):
    """Split indexed modes at a prefix product, retaining original positions."""
    modes, prefix, remaining = tuple(modes), [], count
    for index, (position, width) in enumerate(modes):
        if remaining == 1:
            return tuple(prefix), modes[index:]
        if remaining % width == 0:
            prefix.append((position, width))
            remaining //= width
        elif width % remaining == 0 and position not in indivisible:
            return (*prefix, (position, remaining)), (
                (position, width // remaining),
                *modes[index + 1 :],
            )
        else:
            raise ValueError(f"repeat is not leftmost at mode {position} (extent {width})")
    if remaining != 1:
        raise ValueError(f"leading modes do not contain repeat {count}")
    return tuple(prefix), ()


def shape_div(a, b):
    """CuTe ``shape_div``: divide shapes, folding a scalar over tuple modes."""
    from .utils import static_dim_value  # noqa: PLC0415

    if isinstance(a, tuple):
        if isinstance(b, tuple):
            return tuple(shape_div(x, y) for x, y in zip(a, b, strict=True))
        result = []
        for extent in a:
            result.append(shape_div(extent, b))
            b = shape_div(b, extent)
        return tuple(result)
    if isinstance(b, tuple):
        return shape_div(a, product(b))
    x, y = static_dim_value(a), static_dim_value(b)
    if x is not None and y is not None:
        if x % y and y % x:
            raise ValueError(f"shape_div: {x} and {y} do not divide one another")
        return (x + y - 1) // y
    if compatible(a, b):
        return 1
    raise ValueError(f"shape_div: cannot divide symbolic extents {a!r} and {b!r}")


def compatible(a, b) -> bool:
    """CuTe ``compatible``: each leaf of a equals its corresponding size in b."""
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(
            compatible(x, y) for x, y in zip(a, b, strict=True)
        )
    from ..isl_interop import normalize_dim  # noqa: PLC0415
    from .dim import DimSub, simplify_dim  # noqa: PLC0415

    b = product(b) if isinstance(b, tuple) else b
    if a is b or a == b:
        return True
    try:
        return normalize_dim(simplify_dim(DimSub, (a, b))) == 0
    except (TypeError, ValueError, NotImplementedError):
        return False


def compatible_profile(flat_shape, shape):
    """Nest authored modes into logical axes without splitting or coalescing."""
    from .dim import DimMul, simplify_dim  # noqa: PLC0415
    from .utils import static_dim_value  # noqa: PLC0415

    if any(isinstance(mode, tuple) for mode in flat_shape) and compatible(shape, flat_shape):
        return flat_shape
    extents = flatten(flat_shape)
    groups, cursor = [], 0
    try:
        for extent in shape:
            group, accumulated = [], 1
            if static_dim_value(extent) == 1 and cursor < len(extents) and static_dim_value(extents[cursor]) == 1:
                group.append(extents[cursor])
                cursor += 1
            while not compatible(extent, accumulated):
                mode = extents[cursor]
                left, right = static_dim_value(accumulated), static_dim_value(mode)
                accumulated = left * right if left is not None and right is not None else (
                    mode if accumulated == 1 else simplify_dim(DimMul, (accumulated, mode))
                )
                a, b = static_dim_value(accumulated), static_dim_value(extent)
                if a is not None and b is not None and a > b:
                    raise ValueError("mode crosses a logical axis")
                group.append(mode)
                cursor += 1
            groups.append(tuple(group))
        if any(static_dim_value(mode) != 1 for mode in extents[cursor:]):
            raise ValueError("unassigned layout modes")
        if groups:
            groups[-1] += extents[cursor:]
    except (IndexError, ValueError) as error:
        raise ValueError(f"layout positions {extents} cross logical axes of {tuple(shape)}") from error
    return tuple(group[0] if len(group) == 1 else group for group in groups)


def axis_tiler(profile, sizes, *, steps=None, splits=()):
    """Build row-major per-mode tiles without splitting a placement mode."""
    from .layout import Layout  # noqa: PLC0415
    from .utils import static_dim_value  # noqa: PLC0415

    position = 0

    def tile(mode, span, step):
        nonlocal position
        extents = flatten(mode)
        tiles = [None] * len(extents)
        for index in reversed(range(len(extents))):
            extent = extents[index]
            if index:
                skipped = shape_div(extent, step)
                mode_step = shape_div(extent, skipped)
                width = shape_div(extent, shape_div(extent, span * step))
                count = shape_div(width, mode_step)
            else:
                count, mode_step = span, step
                n, c = static_dim_value(extent), static_dim_value(count)
                if n is not None and c is not None and c and (c - 1) * mode_step >= n:
                    raise ValueError(f"tile exceeds outermost mode {position} (extent {extent})")
            if position + index in splits and not (compatible(count, 1) or compatible(count, extent)):
                raise ValueError(f"tile splits placement mode {position + index} (extent {extent})")
            tiles[index] = None if compatible(count, extent) and mode_step == 1 else Layout((count,), (mode_step,))
            if index:
                span = shape_div(span, count)
                step = shape_div(step, extent)
        position += len(extents)
        return unflatten(tuple(tiles), mode)

    result = []
    for mode, count, step in zip(profile, sizes, steps or (1,) * len(profile), strict=True):
        if count is None:
            position += len(flatten(mode))
            result.append(None)
        elif static_dim_value(count) == 0:
            position += len(flatten(mode))
            result.append(Layout((0,), (step,)))
        else:
            result.append(tile(mode, count, step))
    return tuple(result)


def mode_axes(profile) -> tuple[int, ...]:
    """Each flat mode's logical axis, in authored order."""
    return flatten(tuple(repeat_like(mode, axis) for axis, mode in enumerate(profile)))


def congruent(a, b) -> bool:
    """Whether two int tuples have the same nesting, ignoring their leaves."""
    if isinstance(a, tuple) != isinstance(b, tuple):
        return False
    if not isinstance(a, tuple):
        return True
    return len(a) == len(b) and all(congruent(x, y) for x, y in zip(a, b))


def product(t) -> "ShapeDim":
    from .mesh import Topology  # noqa: PLC0415

    result = 1
    for v in flatten(t):
        if isinstance(v, Topology):
            v = v.size
        result *= v
    return result


def repeat_like(profile, value) -> object:
    """CuTe ``repeat_like``: *value* at every leaf, nested like *profile*.

    What a flat tuple has to say about a nested one is said by building it
    against the nesting and flattening that, rather than by working out which
    flat positions each mode covers.
    """
    if not isinstance(profile, tuple):
        return value
    return tuple(repeat_like(item, value) for item in profile)


def _unflatten(flat: tuple, profile) -> tuple:
    """Take *profile*'s worth of *flat*, returning it nested and what is left."""
    if not isinstance(profile, tuple):
        if not flat:
            raise ValueError("unflatten: the profile asks for more modes than the tuple has")
        return flat[0], flat[1:]
    nested: list = []
    for item in profile:
        value, flat = _unflatten(flat, item)
        nested.append(value)
    return tuple(nested), flat


def unflatten(flat: tuple, profile) -> tuple:
    """CuTe ``unflatten``: nest a flat tuple to *profile*'s structure.

    Only *profile*'s nesting is read, never its leaves, so the profile may be
    the grouping itself. ``flatten(unflatten(t, p)) == t``.
    """
    nested, rest = _unflatten(flat, profile)
    if rest:
        raise ValueError(
            f"unflatten: the profile accounts for {len(flat) - len(rest)} of the "
            f"tuple's {len(flat)} modes"
        )
    return nested


__all__ = [
    "IntTuple",
    "compatible",
    "compatible_profile",
    "axis_tiler",
    "mode_axes",
    "shape_div",
    "congruent",
    "flatten",
    "product",
    "partition_modes",
    "repeat_like",
    "unflatten",
]
