"""One logical-axis window derivation for sharded slices and instruction tiles."""

from __future__ import annotations

from dataclasses import replace
from operator import add, floordiv, mod, mul

import isl

from .dim import DimAdd, DimFloorDiv, DimMod, DimMul, simplify_dim
from .int_tuple import flatten
from .layout import ComposedLayout, Layout, resolve_layout
from .layout_correspondence import layout_correspondence
from .shard_layout import Split
from .utils import static_dim_value


def _dim(op, args):
    numbers = tuple(static_dim_value(value) for value in args)
    if None not in numbers:
        return {DimAdd: add, DimFloorDiv: floordiv, DimMod: mod, DimMul: mul}[op](*numbers)
    return simplify_dim(op, args)


def _aligned(start, size, loops, aliases):
    """Prove block alignment over the authored loop domain, without rewriting Exprs."""
    from tilefoundry.ir.isl_interop import dim_to_isl_pw_aff  # noqa: PLC0415

    number = static_dim_value(start)
    if number is not None:
        return number % size == 0
    values = {}
    names = [f"i{index}" for index in range(len(loops))]
    coords = {id(loop[0]): name for loop, name in zip(loops, names, strict=True)}
    coords.update({alias: coords[root] for alias, root in aliases.items() if root in coords})
    domain = isl.set(f"{{ [{', '.join(names)}] }}")
    try:
        for index, (_var, begin, end, step) in enumerate(loops):
            begin = dim_to_isl_pw_aff(begin, values, coords=coords)
            end = dim_to_isl_pw_aff(end, values, coords=coords)
            step = static_dim_value(step)
            if step is None or step <= 0:
                return False
            iv = isl.pw_aff(f"{{ [{', '.join(names)}] -> [{names[index]}] }}")
            domain = domain.intersect(begin.le_set(iv)).intersect(iv.lt_set(end))
            domain = domain.intersect(iv.sub(begin).mod(isl.val(step)).zero_set())
        position = dim_to_isl_pw_aff(start, values, coords=coords)
        return position.mod(isl.val(size)).non_zero_set().intersect(domain).is_empty()
    except (TypeError, ValueError, NotImplementedError, isl.Error):
        return False


def shard_window_layout(source, logical_shape, starts, sizes, strides, *, loops=(), aliases=None):
    """Keep thread ownership while deriving a representable logical window.

    Split axes retain a complete suffix containing every split mode; a mode
    may factor into adjacent row-major digits. Its start must be block aligned.
    Unsplit axes retain the existing single-mode affine window contract.
    Removed prefix coordinates contribute physical offsets, never new owners.
    """
    aliases = {} if aliases is None else aliases
    layout = source.layout
    if isinstance(layout, ComposedLayout):
        result = shard_window_layout(
            replace(source, layout=layout.outer),
            logical_shape,
            starts,
            sizes,
            strides,
            loops=loops,
            aliases=aliases,
        )
        if isinstance(result, ComposedLayout):
            return replace(
                result, inner=layout.inner, offset=_dim(DimAdd, (layout.offset, result.offset))
            )
        return replace(source, layout=replace(layout, outer=result.layout), attrs=result.attrs)
    if not isinstance(layout, Layout):
        raise ValueError("Slice of a ShardLayout requires a primitive underlying Layout")
    correspondence = layout_correspondence(layout.shape, logical_shape)
    if correspondence.groups is None:
        raise ValueError(f"layout {layout!r} crosses logical axes of {logical_shape}")
    steps = tuple(flatten(resolve_layout(layout).strides))
    split = {attr.axis for attr in source.attrs if isinstance(attr, Split)}
    grouped = any(isinstance(mode, tuple) for mode in layout.shape)
    shape_modes, stride_modes, positions = [], [], {}
    offset = 0
    known_offset = True
    for axis, group in enumerate(correspondence.groups):
        start, size, step = starts[axis], sizes[axis], strides[axis]
        distributed = any(position in split for position in group)
        full = size == logical_shape[axis] and static_dim_value(start) == 0 and step == 1
        kept = []
        if full:
            kept = [
                (position, correspondence.extents[position], steps[position]) for position in group
            ]
        elif distributed or len(group) != 1:
            mesh_axis = next(
                (
                    i
                    for i, attr in enumerate(source.attrs)
                    if isinstance(attr, Split) and attr.axis in group
                ),
                0,
            )
            refusal = (
                f"Slice narrows axis {axis}, which mesh axis {mesh_axis} splits; "
                "a window need not align with the mesh division. Slice before placing, "
                f"or reshard to a layout that leaves axis {axis} whole."
            )
            width = static_dim_value(size)
            if (
                step != 1
                or width is None
                or width <= 0
                or not _aligned(start, width, loops, aliases)
            ):
                raise ValueError(refusal)
            remaining = width
            for position in reversed(group):
                extent = static_dim_value(correspondence.extents[position])
                if extent is None or extent <= 0:
                    raise ValueError(refusal)
                if remaining == 1:
                    if position in split and extent != 1:
                        raise ValueError(refusal)
                    if position in split or extent == 1:
                        kept.append((position, extent, steps[position]))
                elif remaining >= extent and remaining % extent == 0:
                    kept.append((position, extent, steps[position]))
                    remaining //= extent
                elif extent % remaining == 0 and position not in split:
                    kept.append((position, remaining, steps[position]))
                    remaining = 1
                else:
                    raise ValueError(refusal)
            if remaining != 1:
                raise ValueError(refusal)
            kept.reverse()
        else:
            if len(group) != 1:
                raise ValueError(
                    f"Slice narrows axis {axis}, whose layout uses positions {list(group)}; "
                    "the window cannot be represented by one layout axis"
                )
            position = group[0]
            kept = [(position, size, _dim(DimMul, (steps[position], step)))]
        if not kept:
            kept = [(None, 1, 0)]
        origin = sum(len(tuple(flatten(mode))) for mode in shape_modes)
        for index, (position, _extent, _stride) in enumerate(kept):
            if position is not None:
                positions[position] = origin + index
        shapes = tuple(item[1] for item in kept)
        physical = tuple(item[2] for item in kept)
        if grouped:
            shape_modes.append(shapes if len(shapes) > 1 else shapes[0])
            stride_modes.append(physical if len(physical) > 1 else physical[0])
        else:
            shape_modes.extend(shapes)
            stride_modes.extend(physical)
        if static_dim_value(start) == 0:
            continue
        if not distributed and (
            static_dim_value(start) is None
            or layout.strides is None
            or any(static_dim_value(steps[position]) is None for position in group)
            or static_dim_value(step) is None
        ):
            known_offset = False
            continue
        below = 1
        for position in reversed(group):
            extent = correspondence.extents[position]
            digit = _dim(DimFloorDiv, (start, below))
            if position != group[0]:
                digit = _dim(DimMod, (digit, extent))
            kept_extent = next(
                (width for original, width, _stride in kept if original == position), None
            )
            if position not in split and (not distributed or kept_extent != extent):
                offset = _dim(DimAdd, (offset, _dim(DimMul, (digit, steps[position]))))
            below = _dim(DimMul, (below, extent))
    attrs = tuple(
        Split(positions[attr.axis]) if isinstance(attr, Split) else attr for attr in source.attrs
    )
    result = replace(
        source,
        layout=Layout(
            tuple(shape_modes), tuple(stride_modes) if layout.strides is not None else None
        ),
        attrs=attrs,
    )
    number = static_dim_value(offset)
    if not known_offset or number == 0:
        return result
    return ComposedLayout(None, number if number is not None else offset, result)
