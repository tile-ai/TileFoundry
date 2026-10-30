"""Reference execution over static mesh participants on one torch device.

The carrier here is an interpreter value, not another distribution type.
Ownership continues to come from the HIR's Mesh and ShardLayout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

from tilefoundry.evaluator.value import EvalError, TensorValue, TupleValue, Value
from tilefoundry.ir.types import Layout, Mesh, TensorType, TupleType
from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.shard_layout import (
    Broadcast,
    Dynamic,
    Partial,
    _local_readback,
    local_layout,
    shard_layout_of,
)
from tilefoundry.utils.lazy import torch


@dataclass(frozen=True)
class MeshValue(Value):
    """One tensor per logical participant, paired with its global HIR type."""

    values: tuple[TensorValue, ...]
    type: TensorType
    mesh: Mesh


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
    if shard.mesh != mesh:
        raise EvalError("mesh evaluation does not support redistribution between different meshes")
    if len(shard.attrs) != len(flatten(mesh.layout).shape):
        raise EvalError("mesh evaluation requires one distribution attribute per mesh axis")
    if any(isinstance(attr, Dynamic) for attr in shard.attrs):
        raise EvalError("mesh evaluation requires Dynamic distribution to be resolved")
    return shard.attrs


def local_type(held, mesh: Mesh):
    """Present existing op handlers with their participant-local logical shape."""
    if isinstance(held, TupleType):
        return TupleType(tuple(local_type(field, mesh) for field in held.fields))
    if not isinstance(held, TensorType):
        return held
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
        if value.mesh != mesh:
            raise EvalError("nested mesh evaluation requires the same participant mesh")
        return value
    if not isinstance(value, TensorValue):
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
    mesh = ctx.mesh or next((found for arg in ctx.args if (found := mesh_of(arg))), None)
    if mesh is None:
        return handler(ctx)
    args = tuple(distribute(arg, mesh) for arg in ctx.args)
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
