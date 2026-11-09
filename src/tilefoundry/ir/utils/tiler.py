"""Per-mode tiles of a placement, without splitting a placement mode."""

from __future__ import annotations

from tilefoundry.ir import expr, types
from tilefoundry.ir.types import int_tuple

__all__ = ["axis_tiler"]


def axis_tiler(profile, sizes, *, steps=None, splits=()):
    """Build row-major per-mode tiles without splitting a placement mode."""
    position = 0

    def tile(mode, span, step):
        nonlocal position
        extents = int_tuple.flatten(mode)
        tiles = [None] * len(extents)
        for index in reversed(range(len(extents))):
            extent = extents[index]
            if index:
                skipped = int_tuple.shape_div(extent, step)
                mode_step = int_tuple.shape_div(extent, skipped)
                width = int_tuple.shape_div(extent, int_tuple.shape_div(extent, span * step))
                count = int_tuple.shape_div(width, mode_step)
            else:
                count, mode_step = span, step
                n, c = expr.static_dim_value(extent), expr.static_dim_value(count)
                if n is not None and c is not None and c and (c - 1) * mode_step >= n:
                    raise ValueError(f"tile exceeds outermost mode {position} (extent {extent})")
            if position + index in splits and not (int_tuple.compatible(count, 1) or int_tuple.compatible(count, extent)):
                raise ValueError(f"tile splits placement mode {position + index} (extent {extent})")
            tiles[index] = types.Layout((count,), (mode_step,))
            if index:
                span = int_tuple.shape_div(span, count)
                step = int_tuple.shape_div(step, extent)
        position += len(extents)
        return int_tuple.unflatten(tuple(tiles), mode)

    result = []
    for mode, count, step in zip(profile, sizes, steps or (1,) * len(profile), strict=True):
        if count is None:
            position += len(int_tuple.flatten(mode))
            result.append(None)
        elif expr.static_dim_value(count) == 0:
            position += len(int_tuple.flatten(mode))
            result.append(types.Layout((0,), (step,)))
        else:
            result.append(tile(mode, count, step))
    return tuple(result)
