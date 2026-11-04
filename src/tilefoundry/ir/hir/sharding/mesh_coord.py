"""The coordinate of the unit a region is running on."""

from __future__ import annotations

from tilefoundry.core import ShardTensor
from tilefoundry.evaluator.registry import register_eval
from tilefoundry.evaluator.value import TensorValue, TupleValue
from tilefoundry.ir.core import Constant, Op, Tuple
from tilefoundry.ir.core.param_def import ParamDef
from tilefoundry.ir.core.register import register_op
from tilefoundry.ir.mesh_scope import covered_by_scope
from tilefoundry.ir.types import DType, Placement, TensorType, TupleType
from tilefoundry.ir.types.layout import Layout, flatten
from tilefoundry.ir.types.mesh import Mesh
from tilefoundry.ir.types.shard_layout import mesh_coords
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.utils.lazy import torch
from tilefoundry.visitor_registry import register_typeinfer
from tilefoundry.visitor_registry.access_relation import (
    readnone_relations,
    register_access_relation,
)
from tilefoundry.visitor_registry.contexts import TypeInferResults


@register_op(dialect='tf', category='sharding')
class MeshCoord(Op):
    """This unit's coordinates along *axes* of *mesh*: a scalar axis gives one, a vector one each.

    Naming a mesh axis and asking which unit this is are two different
    questions. The first fills in a ``Split`` on a layout and leaves no node
    behind; only the second needs an operation, and this is it. So the mesh is a
    value the node carries rather than the scope it sits in, and whether an
    enclosing region binds that mesh stays a rule about the regions. The axes
    are an operand rather than a name so they can be computed, and so the node
    reads like every other Op in this dialect.
    """

    mesh = ParamDef(kind="attribute", annotation=Mesh)
    axes = ParamDef(kind="input")


def literal_axes(axes) -> tuple[int, ...] | None:
    """The axes a literal operand names, or None when any of them is computed."""
    if isinstance(axes, Tuple):
        found = tuple(static_dim_value(axis) for axis in axes.elements)
        return None if None in found else found
    axis = static_dim_value(axes)
    if axis is not None:
        return (axis,)
    if isinstance(axes, Constant) and isinstance(axes.value, (list, tuple)):
        return tuple(int(axis) for axis in axes.value)
    return None


def _axes_shape(axes_type) -> tuple | None:
    """The result shape: () for one axis, (k,) for k of them, None for anything else."""
    if isinstance(axes_type, TupleType):
        scalars = all(
            isinstance(field, TensorType) and not field.shape for field in axes_type.fields
        )
        return (len(axes_type.fields),) if scalars else None
    if isinstance(axes_type, TensorType) and len(axes_type.shape) <= 1:
        return tuple(axes_type.shape)
    return None


@register_typeinfer(MeshCoord)
def _(call: "Call", ctx: "TypeInferContext") -> TypeInferResults:
    """Coordinates are numbers about this unit, so they carry no placement."""
    if not isinstance(call.target.mesh, Mesh):
        ctx.error(call, "MeshCoord.mesh must be a Mesh")
    try:
        covered = ctx.current_mesh is not None and covered_by_scope(call.target.mesh, ctx.current_mesh)
    except ValueError as error:
        ctx.error(call, str(error))
    if not covered:
        ctx.error(call, "MeshCoord.mesh must be bound by the current mesh scope")
    if not call.args:
        ctx.error(call, "missing required input 'axes'")
    held_shape = _axes_shape(ctx.type_of(call.args[0]))
    if held_shape is None:
        ctx.error(call, "MeshCoord.axes must be an i64 scalar, vector, or tuple of scalars")
    shape = flatten(call.target.mesh.layout).shape
    axes = literal_axes(call.args[0])
    for axis in axes or ():
        if not 0 <= axis < len(shape):
            ctx.error(call, f"axis {axis} is out of range for rank-{len(shape)} mesh")
    extents = tuple(shape[axis] for axis in axes) if axes is not None else shape
    bounds = (
        (0, max(extents))
        if extents
        and all(isinstance(extent, int) and not isinstance(extent, bool) for extent in extents)
        else None
    )
    held = (
        TensorType.scalar(DType.i64, storage=StorageKind.RMEM)
        if not held_shape
        else TensorType(
            shape=held_shape, dtype=DType.i64, layout=Layout(held_shape), storage=StorageKind.RMEM
        )
    )
    return TypeInferResults(held, value_range=bounds)


@register_eval(MeshCoord)
def _eval_mesh_coord(ctx):
    """Each placement's coordinates along the axes: a local view over the mesh."""
    axes = ctx.args[0]
    if isinstance(axes, TupleValue):
        axes = torch.stack([element.data.reshape(()) for element in axes.elements])
    else:
        axes = axes.data
    axes = axes.long()
    mesh = ctx.op.mesh
    pieces = tuple(
        (
            placement,
            torch.as_tensor(
                mesh_coords(mesh, placement.program_ids(mesh.topologies)),
                dtype=torch.int64,
                device=axes.device,
            )[axes],
        )
        for placement in Placement.each(mesh)
    )
    return TensorValue(ShardTensor(ctx.result_type, pieces))


register_access_relation(MeshCoord)(readnone_relations(lambda types: ()))


__all__ = ["MeshCoord", "literal_axes"]
