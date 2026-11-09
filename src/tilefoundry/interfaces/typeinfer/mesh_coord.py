"""Type inference of ``MeshCoord``: coordinates are numbers about this unit."""

from __future__ import annotations

from tilefoundry.ir import dim, expr, types, utils

from .context import TypeInferResults
from .registry import register

__all__: list[str] = []


def axes_shape(axes_type) -> tuple | None:
    """The result shape: () for one axis, (k,) for k of them, None for anything else."""
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
    """Coordinates are numbers about this unit, so they carry no placement."""
    if not isinstance(call.target.mesh, types.Mesh):
        ctx.error(call, "MeshCoord.mesh must be a Mesh")
    try:
        covered = ctx.current_mesh is not None and utils.covered_by_scope(call.target.mesh, ctx.current_mesh)
    except ValueError as error:
        ctx.error(call, str(error))
    if not covered:
        ctx.error(call, "MeshCoord.mesh must be bound by the current mesh scope")
    if not call.args:
        ctx.error(call, "missing required input 'axes'")
    held_shape = axes_shape(ctx.type_of(call.args[0]))
    if held_shape is None:
        ctx.error(call, "MeshCoord.axes must be an i64 scalar, vector, or tuple of scalars")
    shape = types.flatten(call.target.mesh.layout).shape
    axes = dim.literal_axes(call.args[0])
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
        types.TensorType.from_scalar(expr.DType.i64, storage=expr.StorageKind.RMEM)
        if not held_shape
        else types.TensorType(
            shape=held_shape, dtype=expr.DType.i64, layout=types.Layout(held_shape), storage=expr.StorageKind.RMEM
        )
    )
    return TypeInferResults(held, value_range=bounds)
