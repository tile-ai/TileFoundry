"""Which part of a sharded tensor one mesh instance holds.

The same two answers under the same two names as the device side
(layout/shard_layout.cuh): ``local_layout`` is what an instance holds and
``local_layout_and_offset`` adds where its part begins. The host states the
tensor shape as well, because ``canonical_shard_layout`` factors a split axis
away from the tensor's own rank, and the ids as well, because a level the
placement leaves unfixed is the device's to divide rather than this side's.

See [shard §7.6](docs/spec/shard.md#76-local_layout).
"""

from __future__ import annotations

from collections.abc import Iterable

from .int_tuple import flatten
from .layout import Layout
from .layout_algebra import c_order_strides, idx2crd
from .mesh import positions_at, topology_axes
from .shard_layout import ShardLayout, layout_axis_to_tensor_axis, split_target_axes


def _extents(shard: ShardLayout) -> tuple[int, ...]:
    """Each mesh axis's extent, flat and in the order the attrs index them."""
    values = tuple(flatten(shard.mesh.layout.shape))
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"local: this mesh states a symbolic extent {value!r}; which part is "
                f"one instance's is a number, so bind the mesh first"
            )
    return values


def _cutting(shard: ShardLayout, tensor_shape: tuple) -> tuple[int | None, ...]:
    """Per mesh axis, the tensor axis its attr cuts, or ``None`` when none."""
    return split_target_axes(shard, tuple(tensor_shape))


def _strides(shard: ShardLayout, tensor_shape: tuple) -> tuple[int, ...]:
    """Each tensor axis's own stride, read off the factored layout.

    A split axis is stated as mesh-sized positions plus a residual, so the
    step of the tensor axis itself is the last of them; a layout that
    materialized no strides is the C order its canonical form would have.
    """
    fallback = c_order_strides(tuple(tensor_shape))
    stated = shard.layout.strides
    if stated is None:
        return fallback
    axes = layout_axis_to_tensor_axis(shard.layout.shape, tuple(tensor_shape))
    found = dict(zip(axes, stated, strict=False))
    return tuple(found.get(axis, fallback[axis]) for axis in range(len(tensor_shape)))


def _narrowed(
    shard: ShardLayout, tensor_shape: tuple, axis: int, dividing: Iterable[int]
) -> int:
    """Tensor *axis* divided by each mesh axis in *dividing* that cuts it.

    Several may. Two axes of one level cut a tensor axis into a grid, and so
    do two levels, one taking a block of what the other left; each division is
    of what the previous one left. An extent its mesh axis does not divide is
    refused by name: a shard would then not be one slice.
    """
    extents = _extents(shard)
    allowed = set(dividing)
    held = tensor_shape[axis]
    for mesh_axis, target in enumerate(_cutting(shard, tensor_shape)):
        if target != axis or mesh_axis >= len(extents) or mesh_axis not in allowed:
            continue
        count = extents[mesh_axis]
        if held % count:
            raise ValueError(
                f"local: axis {axis} has extent {held}, which its mesh axis of "
                f"{count} positions does not divide; a shard would not be one slice"
            )
        held //= count
    return held


def _inner(shard: ShardLayout, tensor_shape: tuple, mesh_axis: int, axis: int) -> int:
    """How much of tensor *axis* one step of *mesh_axis* steps over.

    The axes cutting one tensor axis are ordered outermost first, so a step of
    one of them clears everything the axes inside it hold.
    """
    extents = _extents(shard)
    product = 1
    for other, target in enumerate(_cutting(shard, tensor_shape)):
        if other > mesh_axis and target == axis and other < len(extents):
            product *= extents[other]
    return product


def _stride(shard: ShardLayout, tensor_shape: tuple, mesh_axis: int, axis: int) -> int:
    """What one step of *mesh_axis* costs the slice's origin, in elements.

    As the device's ``stride<Ax>(sl)`` has it: the extent one instance is left
    of the tensor axis, times what the axes inside this one hold of it, times
    the tensor axis's own stride. Every axis cutting it divides here, unfixed
    or not, because a step must clear what the device will divide too.
    """
    return (
        _narrowed(shard, tensor_shape, axis, range(len(_extents(shard))))
        * _inner(shard, tensor_shape, mesh_axis, axis)
        * _strides(shard, tensor_shape)[axis]
    )


def _positions(shard: ShardLayout, ids: tuple[int | None, ...]) -> dict[int, int]:
    """Each mesh axis's coordinate, for the levels an id was given for.

    A level with no id is left unfixed and names no coordinate, so the axes it
    owns divide nothing and the whole of them stays.
    """
    names = tuple(topology.name for topology in shard.mesh.topologies)
    found: dict[int, int] = {}
    for index, mesh_axes in enumerate(topology_axes(shard.mesh)):
        program_id = ids[index] if index < len(ids) else None
        if program_id is None:
            continue
        coord = idx2crd(program_id, *positions_at(shard.mesh, names[index]))
        for mesh_axis, position in zip(mesh_axes, coord, strict=True):
            found[mesh_axis] = position
    return found


def local_layout(
    shard: ShardLayout, tensor_shape: tuple, ids: tuple[int | None, ...]
) -> Layout:
    """What one instance holds: each tensor axis narrowed by what cuts it.

    Narrowed by the axes *ids* names, and no others: an axis nothing cuts is
    held whole, and so is one cut only by a level the placement left unfixed --
    ``cta`` and ``thread`` are the device's to divide. The strides stay the
    whole tensor's, as they do on the device: a slice of it is the same rows,
    the same distance apart.
    """
    dividing = _positions(shard, ids).keys()
    return Layout(
        shape=tuple(
            _narrowed(shard, tensor_shape, axis, dividing)
            for axis in range(len(tensor_shape))
        ),
        strides=_strides(shard, tensor_shape),
    )


def local_layout_and_offset(
    shard: ShardLayout, tensor_shape: tuple, ids: tuple[int | None, ...]
) -> tuple[Layout, int]:
    """That layout, and how far into the tensor this instance's part begins.

    What ``cute::slice_and_offset`` is to a Layout, as on the device: the
    offset is the instance's mesh coordinate dotted with one stride per mesh
    axis. It is counted in the strides ``local_layout`` reports, so a reader
    laid out otherwise would read the wrong elements.
    """
    cutting = _cutting(shard, tensor_shape)
    offset = 0
    for mesh_axis, position in _positions(shard, ids).items():
        axis = cutting[mesh_axis] if mesh_axis < len(cutting) else None
        if axis is None:
            continue
        offset += position * _stride(shard, tensor_shape, mesh_axis, axis)
    return local_layout(shard, tensor_shape, ids), offset


__all__ = ["local_layout", "local_layout_and_offset"]
