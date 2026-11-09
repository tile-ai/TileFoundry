"""Evaluation of ``MeshCoord``: each placement's coordinates along the axes."""

from __future__ import annotations

from tilefoundry import core, utils
from tilefoundry.ir import dim, types

from .registry import register
from .value import TensorValue, TupleValue

__all__: list[str] = []


@register(dim.MeshCoord)
def evaluate_mesh_coord(ctx):
    """Each placement's coordinates along the axes: a local view over the mesh."""
    axes = ctx.args[0]
    if isinstance(axes, TupleValue):
        axes = utils.torch.stack([element.data.reshape(()) for element in axes.elements])
    else:
        axes = axes.data
    axes = axes.long()
    mesh = ctx.op.mesh
    pieces = tuple(
        (
            placement,
            utils.torch.as_tensor(
                types.mesh_coords(mesh, placement.program_ids(mesh.topologies)),
                dtype=utils.torch.int64,
                device=axes.device,
            )[axes],
        )
        for placement in types.Placement.from_mesh(mesh)
    )
    return TensorValue(core.ShardTensor(ctx.result_type, pieces))
