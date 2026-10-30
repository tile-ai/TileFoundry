"""Reference execution over static mesh participants on one torch device.

The carrier here is an interpreter value, not another distribution type.
Ownership continues to come from the HIR's Mesh and ShardLayout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from math import prod

from tilefoundry.evaluator.value import EvalError, TensorValue, TupleValue, Value
from tilefoundry.ir.types import Layout, Mesh, TensorType, TupleType
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    Dynamic,
    Partial,
    Split,
    _local_readback,
    local_layout,
    shard_layout_of,
    split_target_axes,
)
from tilefoundry.utils.lazy import torch


@dataclass(frozen=True)
class MeshValue(Value):
    """One tensor per logical participant, paired with its global HIR type."""

    values: tuple[TensorValue, ...]
    type: TensorType
    mesh: Mesh


def participant_mesh(mesh: Mesh) -> bool:
    """Device participation is distinct from existing in-device reference views."""
    return len(mesh.topologies) == 1 and mesh.topologies[0].name == "gpu"


def same_participants(left: Mesh, right: Mesh) -> bool:
    """Axis names label a mesh; topology and arrangement identify its units."""
    return left.topologies == right.topologies and left.layout == right.layout


@lru_cache(maxsize=256)
def coordinates(mesh: Mesh) -> tuple[tuple[int, ...], ...]:
    """Enumerate logical coordinates without assuming physical rank order."""
    shape = flatten(mesh.layout).shape
    if not shape or any(type(extent) is not int or extent <= 0 for extent in shape):
        raise EvalError("mesh evaluation requires positive static mesh extents")
    return tuple(product(*(range(extent) for extent in shape)))


def attributes(held: TensorType, mesh: Mesh):
    """Read a distribution in the active mesh, refusing unresolved decisions."""
    shard = shard_layout_of(held.layout)
    if shard is None:
        return (Broadcast(),) * len(flatten(mesh.layout).shape)
    if not same_participants(shard.mesh, mesh):
        raise EvalError("mesh evaluation does not support redistribution between different meshes")
    if len(shard.attrs) != len(flatten(mesh.layout).shape):
        raise EvalError("mesh evaluation requires one distribution attribute per mesh axis")
    if any(isinstance(attr, Dynamic) for attr in shard.attrs):
        raise EvalError("mesh evaluation requires Dynamic distribution to be resolved")
    return shard.attrs


@lru_cache(maxsize=256)
def local_type(held, mesh: Mesh):
    """Present existing op handlers with their participant-local logical shape."""
    if isinstance(held, TupleType):
        return TupleType(tuple(local_type(field, mesh) for field in held.fields))
    if not isinstance(held, TensorType):
        return held
    if any(type(extent) is not int for extent in held.shape):
        raise EvalError("GPU participant evaluation requires concrete tensor shapes; specialize first")
    attributes(held, mesh)
    shard = shard_layout_of(held.layout)
    if shard is None:
        return held
    layout, _ = local_layout(shard, held.shape, (0,) * len(shard.attrs))
    shape = _local_readback(layout).shape
    return replace(held, shape=shape, layout=Layout(shape))


def distribute(value: Value, mesh: Mesh) -> Value:
    """Slice invariant global inputs once; never manufacture Partial copies."""
    from tilefoundry.runtime.tensor import logical_local_view  # noqa: PLC0415 -- import cycle

    if isinstance(value, TupleValue):
        return TupleValue(tuple(distribute(item, mesh) for item in value.elements))
    if isinstance(value, MeshValue):
        if not same_participants(value.mesh, mesh):
            raise EvalError("nested mesh evaluation requires the same participant mesh")
        return value
    if not participant_mesh(mesh):
        return value
    if not isinstance(value, TensorValue):
        return value
    shard = shard_layout_of(value.type.layout)
    if shard is None or all(isinstance(attr, Broadcast) for attr in shard.attrs):
        return value
    attrs = attributes(value.type, mesh)
    if any(isinstance(attr, Partial) for attr in attrs):
        raise EvalError("a canonical input cannot supply unresolved Partial contributions")
    held = local_type(value.type, mesh)
    return MeshValue(
        tuple(
            TensorValue(logical_local_view(value.data, value.type, coordinate), held)
            for coordinate in coordinates(mesh)
        ),
        value.type,
        mesh,
    )


def participant(value, index: int):
    """Select one execution's operands, preserving tuple structure."""
    if isinstance(value, MeshValue):
        return value.values[index]
    if isinstance(value, TupleValue):
        return TupleValue(tuple(participant(item, index) for item in value.elements))
    return value


def mesh_of(value):
    """Find participation inside an operand or a tuple operand."""
    if isinstance(value, MeshValue):
        return value.mesh
    if isinstance(value, TupleValue):
        return next((found for item in value.elements if (found := mesh_of(item))), None)
    return None


def pack(values, held, mesh: Mesh):
    """Reassemble per-op results without losing their declared global type."""
    if isinstance(values[0], TupleValue):
        return TupleValue(tuple(
            pack(tuple(value.elements[index] for value in values), item, mesh)
            for index, item in enumerate(held.fields)
        ))
    for value in values:
        if tuple(value.data.shape) != tuple(local_type(held, mesh).shape):
            raise EvalError(
                f"participant result shape {tuple(value.data.shape)} disagrees with "
                f"the declared local shape {tuple(local_type(held, mesh).shape)}"
            )
    return MeshValue(tuple(values), held, mesh)


def observe(value: MeshValue):
    """Reconstruct a logical output; this is observation, not a program collective."""
    from tilefoundry.runtime.tensor import logical_local_view  # noqa: PLC0415 -- import cycle

    attrs = attributes(value.type, value.mesh)
    if any(isinstance(attr, Partial) for attr in attrs):
        raise EvalError("cannot observe unresolved Partial; insert an explicit Reshard")
    first = value.values[0].data
    result = torch.empty(value.type.shape, dtype=first.dtype, device=first.device)
    covered = torch.zeros(value.type.shape, dtype=torch.bool, device=first.device)
    for coordinate, local in zip(coordinates(value.mesh), value.values, strict=True):
        target = logical_local_view(result, value.type, coordinate)
        written = logical_local_view(covered, value.type, coordinate)
        if written.any() and not torch.equal(target[written], local.data[written]):
            raise EvalError("participant values disagree on a Broadcast/plain output")
        target.copy_(local.data)
        written.fill_(True)
    if not covered.all():
        raise EvalError("mesh output does not cover its declared logical tensor")
    return result


def evaluate_op(handler, ctx):
    """Execute an ordinary op once per participant with its own local values."""
    from tilefoundry.ir.hir.sharding.mesh_coord import MeshCoord  # noqa: PLC0415 -- import cycle
    from tilefoundry.ir.hir.sharding.reshard import Reshard  # noqa: PLC0415 -- import cycle
    from tilefoundry.ir.hir.tensor.reduce import Reduce  # noqa: PLC0415 -- import cycle

    if isinstance(ctx.op, Reshard):
        return handler(ctx)
    mesh = ctx.mesh or next((found for arg in ctx.args if (found := mesh_of(arg))), None)
    if mesh is None or not participant_mesh(mesh):
        return handler(ctx)
    if not isinstance(ctx.op, MeshCoord) and not any(mesh_of(arg) for arg in ctx.args):
        return distribute(handler(ctx), mesh)
    args = tuple(distribute(arg, mesh) for arg in ctx.args)
    if isinstance(ctx.op, Reduce):
        source = args[0]
        shard = shard_layout_of(source.type.layout)
        axes = tuple(axis % len(source.type.shape) for axis in ctx.op.axes)
        if shard is not None and any(axis in axes for axis in split_target_axes(shard, source.type.shape)):
            complete = TensorValue(observe(source), source.type)
            result = handler(replace(ctx, args=(complete,), mesh=None, coordinates=None))
            return distribute(result, mesh)
    for arg in args:
        if isinstance(arg, MeshValue) and any(
            isinstance(attr, Partial) for attr in attributes(arg.type, mesh)
        ):
            raise EvalError("resolve Partial with Reshard before participant-local operations")
    results = tuple(
        handler(replace(
            ctx,
            mesh=mesh,
            coordinates=coordinate,
            args=tuple(participant(arg, index) for arg in args),
            result_type=local_type(ctx.result_type, mesh),
        ))
        for index, coordinate in enumerate(coordinates(mesh))
    )
    return pack(results, ctx.result_type, mesh)


def redistribute(value, held: TensorType, mesh: Mesh | None):
    """Resolve explicit logical transfers using actual participant contributions.

    Groups keep every unchanged mesh coordinate fixed. Within a group, Split
    pieces reconstruct indices, Broadcast copies agree, and distinct Partial
    coordinates contribute once to a sum. No network schedule is implied.
    """
    from tilefoundry.runtime.tensor import logical_local_view  # noqa: PLC0415 -- import cycle

    target_shard = shard_layout_of(held.layout)
    mesh = mesh or mesh_of(value) or (target_shard.mesh if target_shard else None)
    if mesh is None or not participant_mesh(mesh):
        if isinstance(value, MeshValue):
            raise EvalError("nested mesh evaluation requires the same participant mesh")
        return TensorValue(value.data, held)
    value = distribute(value, mesh)
    before, after = attributes(value.type, mesh), attributes(held, mesh)
    if isinstance(value, TensorValue):
        if all(isinstance(attr, Broadcast) for attr in (*before, *after)):
            return TensorValue(value.data, held)
        value = MeshValue(
            (TensorValue(value.data, local_type(value.type, mesh)),) * len(coordinates(mesh)),
            value.type, mesh,
        )
    changed = tuple(axis for axis, pair in enumerate(zip(before, after, strict=True)) if pair[0] != pair[1])
    for axis in changed:
        source, target = before[axis], after[axis]
        supported = (
            isinstance(source, Broadcast) and isinstance(target, Split)
            or isinstance(source, Split) and isinstance(target, Broadcast)
            or isinstance(source, Partial) and source.reduction == "sum"
            and isinstance(target, (Split, Broadcast))
        )
        if not supported:
            raise EvalError(f"unsupported redistribution on mesh axis {axis}: {source} -> {target}")
    coords = coordinates(mesh)
    result_type = local_type(held, mesh)
    if not changed:
        indices = torch.arange(prod(value.type.shape)).reshape(value.type.shape)
        if any(not torch.equal(
            logical_local_view(indices, value.type, coordinate),
            logical_local_view(indices, held, coordinate),
        ) for coordinate in coords):
            raise EvalError("unsupported Split-to-Split redistribution")
        return MeshValue(
            tuple(TensorValue(item.data, result_type) for item in value.values), held, mesh,
        )
    groups = {}
    for index, coordinate in enumerate(coords):
        key = tuple(at for axis, at in enumerate(coordinate) if axis not in changed)
        groups.setdefault(key, []).append(index)
    outputs = [None] * len(coords)
    partial_axes = tuple(axis for axis in changed if isinstance(before[axis], Partial))
    first = value.values[0].data
    for members in groups.values():
        contributions = {}
        for index in members:
            coordinate = coords[index]
            key = tuple(coordinate[axis] for axis in partial_axes)
            if key not in contributions:
                contributions[key] = (
                    torch.zeros(value.type.shape, dtype=first.dtype, device=first.device),
                    torch.zeros(value.type.shape, dtype=torch.bool, device=first.device),
                )
            full, covered = contributions[key]
            target = logical_local_view(full, value.type, coordinate)
            written = logical_local_view(covered, value.type, coordinate)
            data = value.values[index].data
            if written.any() and not torch.equal(target[written], data[written]):
                raise EvalError("redistribution found disagreeing Broadcast contributions")
            target.copy_(data)
            written.fill_(True)
        full = sum(item[0] for item in contributions.values())
        for index in members:
            if any(not logical_local_view(covered, held, coords[index]).all()
                   for _, covered in contributions.values()):
                raise EvalError("redistribution cannot cover the destination shard")
            data = logical_local_view(full, held, coords[index]).clone()
            outputs[index] = TensorValue(data, result_type)
    return MeshValue(tuple(outputs), held, mesh)
