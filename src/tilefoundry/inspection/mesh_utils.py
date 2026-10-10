"""Recover lexical mesh selections for source inspection."""

from __future__ import annotations

from math import prod

from tilefoundry.ir.types.layout import ComposedLayout, Layout, flatten
from tilefoundry.ir.types.layout import size as layout_size
from tilefoundry.ir.types.layout_algebra import composition, left_inverse
from tilefoundry.ir.types.mesh import Mesh, levels, starts
from tilefoundry.ir.types.stride import compact_row_major, crd2idx, idx2crd


def sub_box(parent: Mesh, mesh: Mesh) -> tuple[slice, ...] | None:
    """Decode the first and last physical positions into a parent's sub-box."""
    if parent.topologies != mesh.topologies or not isinstance(parent.layout, Layout):
        return None
    if not isinstance(mesh.layout, ComposedLayout) or mesh.layout.inner is not None:
        return None
    if not isinstance(mesh.layout.outer, Layout):
        return None
    source, target = flatten(parent.layout), flatten(mesh.layout.outer)
    strides = source.strides or compact_row_major(source.shape)
    target_strides = target.strides or compact_row_major(target.shape)
    offset = mesh.layout.offset
    if not all(isinstance(one, int) for one in (*source.shape, *target.shape, *strides, *target_strides, offset)):
        return None
    if any(step <= 0 for step in strides) or any(step < 0 for step in target_strides):
        return None
    last = offset + sum((extent - 1) * step for extent, step in zip(target.shape, target_strides))
    lower = idx2crd(offset, source.shape, strides)
    upper = idx2crd(last, source.shape, strides)
    if any(start > stop for start, stop in zip(lower, upper)):
        return None
    if crd2idx(lower, source.shape, strides) != offset or crd2idx(upper, source.shape, strides) != last:
        return None
    if prod(stop - start + 1 for start, stop in zip(lower, upper)) != layout_size(target):
        return None
    return tuple(slice(start, stop + 1) for start, stop in zip(lower, upper))


def selection_layout(mesh: Mesh, selection: Mesh) -> Layout | None:
    """Invert reversed selection modes so local indices use row-major numbering.

    Composition uses AssertionError to reject indivisible mode strides.
    """
    if mesh.topologies != selection.topologies or len(mesh.topologies) != 1:
        return None
    if not isinstance(mesh.layout, ComposedLayout) or mesh.layout.inner is not None:
        return None
    if not isinstance(mesh.layout.outer, Layout) or mesh.layout.offset != starts(selection)[0]:
        return None
    source = flatten(levels(selection)[0])
    target = flatten(mesh.layout.outer)
    if layout_size(source) != layout_size(target):
        return None
    steps = source.strides or compact_row_major(source.shape)
    if not all(isinstance(one, int) for one in (*source.shape, *steps, *target.shape, *(target.strides or compact_row_major(target.shape)))):
        return None
    reversed_source = Layout(tuple(reversed(source.shape)), tuple(reversed(steps)))
    try:
        local = flatten(composition(left_inverse(reversed_source), target))
        order = compact_row_major(local.shape)
        local = Layout(local.shape, tuple(
            order[axis] if extent == 1 else step
            for axis, (extent, step) in enumerate(zip(local.shape, local.strides))
        ))
        if composition(source, local, major="row") != target:
            return None
    except (AssertionError, TypeError, ValueError):
        return None
    return local
