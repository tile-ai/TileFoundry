"""Which part of a sharded tensor one mesh instance holds.

The same algebra the device side runs in ``local_layout_and_offset``
(layout/shard_layout.cuh): a tensor axis is divided by every mesh axis that
cuts it, and one step of such an axis clears everything the axes inside it
hold. What each side does with the answer differs -- a kernel dots it with the
shard layout's own strides, a host slices a ``torch.Tensor`` with strides of
its own -- so what is shared is the window, not the offset.

See [shard §7](docs/spec/shard.md#7-shardlayout).
"""

from __future__ import annotations

from .int_tuple import flatten
from .layout_algebra import idx2crd
from .mesh import level_axes, positions_at
from .shard_layout import ShardLayout, split_target_axes


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


def divisor(shard: ShardLayout, tensor_shape: tuple, axis: int) -> int:
    """What tensor *axis* is divided by: every mesh axis that cuts it.

    Several may. Two axes of one level cut a tensor axis into a grid, and so do
    two levels, one taking a block of what the other left. Each division is of
    what the previous one left, so their extents multiply.
    """
    extents = _extents(shard)
    product = 1
    for mesh_axis, target in enumerate(_cutting(shard, tensor_shape)):
        if target == axis and mesh_axis < len(extents):
            product *= extents[mesh_axis]
    return product


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


def _positions(shard: ShardLayout, ids: tuple[int | None, ...]) -> dict[int, int]:
    """Each mesh axis's coordinate, for the levels an id was given for.

    A level with no id is left unfixed and names no coordinate, so the axes it
    owns divide nothing and the whole of them stays.
    """
    names = tuple(topology.name for topology in shard.mesh.topologies)
    found: dict[int, int] = {}
    for index, mesh_axes in enumerate(level_axes(shard.mesh)):
        program_id = ids[index] if index < len(ids) else None
        if program_id is None:
            continue
        coord = idx2crd(program_id, *positions_at(shard.mesh, names[index]))
        for mesh_axis, position in zip(mesh_axes, coord, strict=True):
            found[mesh_axis] = position
    return found


def local_window(
    shard: ShardLayout, tensor_shape: tuple, ids: tuple[int | None, ...]
) -> tuple[tuple[int, int], ...]:
    """Per tensor axis, the ``(start, extent)`` this instance holds.

    An axis nothing cuts is held whole, and so is one cut only by a level the
    placement left unfixed -- ``cta`` and ``thread`` are the device's to
    divide. An extent a mesh axis does not divide is refused by name: a shard
    would then not be one slice.
    """
    extents = _extents(shard)
    cutting = _cutting(shard, tensor_shape)
    positions = _positions(shard, ids)
    window: list[tuple[int, int]] = []
    for axis, whole in enumerate(tensor_shape):
        held = whole
        start = 0
        for mesh_axis, target in enumerate(cutting):
            if target != axis or mesh_axis >= len(extents):
                continue
            count = extents[mesh_axis]
            if mesh_axis not in positions:
                continue
            if held % count:
                raise ValueError(
                    f"local: axis {axis} has extent {held}, which its mesh axis of "
                    f"{count} positions does not divide; a shard would not be one slice"
                )
            held //= count
            start += positions[mesh_axis] * held * _inner(
                shard, tensor_shape, mesh_axis, axis
            )
        window.append((start, held))
    return tuple(window)


__all__ = ["divisor", "local_window"]
