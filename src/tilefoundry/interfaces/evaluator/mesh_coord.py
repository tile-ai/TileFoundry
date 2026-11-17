"""Evaluation of ``MeshCoord``: each placement's coordinates along the mesh axes."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.ir import dim, types

from .registry import register
from .value import TensorValue, TupleValue

__all__: list[str] = []


@register(dim.MeshCoord)
def evaluate_mesh_coord(ctx):
    """Each placement's coordinates along the mesh axes: a local view over the mesh."""
    mesh_axes = ctx.args[0]
    if isinstance(mesh_axes, TupleValue):
        mesh_axes = utils.torch.stack([element.data.reshape(()) for element in mesh_axes.elements])
    else:
        mesh_axes = mesh_axes.data
    mesh_axes = mesh_axes.long()
    mesh = ctx.op.mesh
    pieces = tuple(
        (
            placement,
            utils.torch.as_tensor(
                placement.coords(mesh),
                dtype=utils.torch.int64,
                device=mesh_axes.device,
            )[mesh_axes],
        )
        for placement in types.Placement.from_mesh(mesh)
    )
    return TensorValue(core.ShardTensor(ctx.result_type, pieces))
