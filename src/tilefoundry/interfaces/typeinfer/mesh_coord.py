"""Type inference of ``MeshCoord``: coordinates are numbers about this unit."""

from __future__ import annotations

from tilefoundry.ir import dim, expr, types, utils

from .context import TypeInferResults
from .registry import register

__all__: list[str] = []


def axes_shape(axes_type) -> tuple | None:
    """The result shape: () for one mesh axis, (k,) for k of them, None for anything else."""
    if isinstance(axes_type, types.TupleType):
        scalars = all(
            isinstance(field, types.TensorType) and not field.shape for field in axes_type.fields
        )
        return (len(axes_type.fields),) if scalars else None
    if isinstance(axes_type, types.TensorType) and len(axes_type.shape) <= 1:
        return tuple(axes_type.shape)
    return None


@register(dim.MeshCoord)
def infer_mesh_coord(call: "Call", ctx: "TypeInferContext") -> TypeInferResults:
    """Coordinates are numbers about this unit: Varying on each mesh axis they read."""
    if not isinstance(call.target.mesh, types.Mesh):
        ctx.error(call, "MeshCoord.mesh must be a Mesh")
    try:
        covered = ctx.mesh_scope is not None and utils.equals_mesh_scope(call.target.mesh, ctx.mesh_scope)
    except ValueError as error:
        ctx.error(call, str(error))
    if not covered:
        ctx.error(call, "MeshCoord.mesh must be bound by the current mesh scope")
    if not call.args:
        ctx.error(call, "missing required input 'axes'")
    held_shape = axes_shape(ctx.type_of(call.args[0]))
    if held_shape is None:
        ctx.error(call, "MeshCoord.axes must be an i64 scalar, vector, or tuple of scalars")
    shape = call.target.mesh.shape
    mesh_axes = dim.literal_axes(call.args[0])
    for mesh_axis in mesh_axes or ():
        if not 0 <= mesh_axis < len(shape):
            ctx.error(call, f"mesh axis {mesh_axis} is out of range for rank-{len(shape)} mesh")
    extents = tuple(shape[mesh_axis] for mesh_axis in mesh_axes) if mesh_axes is not None else shape
    bounds = (
        (0, max(extents))
        if extents
        and all(isinstance(extent, int) and not isinstance(extent, bool) for extent in extents)
        else None
    )
    read = range(len(shape)) if mesh_axes is None else mesh_axes
    attrs = tuple(
        types.Varying() if mesh_axis in read else types.Broadcast()
        for mesh_axis in range(len(shape))
    )
    held = types.TensorType(
        shape=held_shape,
        dtype=expr.DType.i64,
        layout=types.ShardLayout(types.Layout(held_shape), attrs, call.target.mesh),
        storage=expr.StorageKind.RMEM,
    )
    return TypeInferResults(held, value_range=bounds)
