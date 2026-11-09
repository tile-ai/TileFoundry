"""Recover lexical mesh selections for source inspection."""

from __future__ import annotations

from math import prod

from tilefoundry.ir import types
from tilefoundry.ir.types import layout_algebra


def sub_box(parent: types.Mesh, mesh: types.Mesh) -> tuple[slice, ...] | None:
    """Decode the first and last physical positions into a parent's sub-box."""
    if parent.topologies != mesh.topologies or not isinstance(parent.layout, types.Layout):
        return None
    if not isinstance(mesh.layout, types.ComposedLayout) or mesh.layout.inner is not None:
        return None
    if not isinstance(mesh.layout.outer, types.Layout):
        return None
    source, target = types.flatten(parent.layout), types.flatten(mesh.layout.outer)
    strides = source.strides or types.compact_row_major(source.shape)
    target_strides = target.strides or types.compact_row_major(target.shape)
    offset = mesh.layout.offset
    if not all(isinstance(one, int) for one in (*source.shape, *target.shape, *strides, *target_strides, offset)):
        return None
    if any(step <= 0 for step in strides) or any(step < 0 for step in target_strides):
        return None
    last = offset + sum((extent - 1) * step for extent, step in zip(target.shape, target_strides))
    lower = types.idx2crd(offset, source.shape, strides)
    upper = types.idx2crd(last, source.shape, strides)
    if any(start > stop for start, stop in zip(lower, upper)):
        return None
    if types.crd2idx(lower, source.shape, strides) != offset or types.crd2idx(upper, source.shape, strides) != last:
        return None
    if prod(stop - start + 1 for start, stop in zip(lower, upper)) != types.size(target):
        return None
    return tuple(slice(start, stop + 1) for start, stop in zip(lower, upper))


def selection_layout(mesh: types.Mesh, selection: types.Mesh) -> types.Layout | None:
    """Invert reversed selection modes so local indices use row-major numbering.

    Composition uses AssertionError to reject indivisible mode strides.
    """
    if mesh.topologies != selection.topologies or len(mesh.topologies) != 1:
        return None
    if not isinstance(mesh.layout, types.ComposedLayout) or mesh.layout.inner is not None:
        return None
    if not isinstance(mesh.layout.outer, types.Layout) or mesh.layout.offset != types.starts(selection)[0]:
        return None
    source = types.flatten(types.levels(selection)[0])
    target = types.flatten(mesh.layout.outer)
    if types.size(source) != types.size(target):
        return None
    steps = source.strides or types.compact_row_major(source.shape)
    if not all(isinstance(one, int) for one in (*source.shape, *steps, *target.shape, *(target.strides or types.compact_row_major(target.shape)))):
        return None
    reversed_source = types.Layout(tuple(reversed(source.shape)), tuple(reversed(steps)))
    try:
        local = types.flatten(layout_algebra.composition(layout_algebra.left_inverse(reversed_source), target))
        order = types.compact_row_major(local.shape)
        local = types.Layout(local.shape, tuple(
            order[axis] if extent == 1 else step
            for axis, (extent, step) in enumerate(zip(local.shape, local.strides))
        ))
        if layout_algebra.composition(source, local, major="row") != target:
            return None
    except (AssertionError, TypeError, ValueError):
        return None
    return local


__all__ = ["selection_layout", "sub_box"]
