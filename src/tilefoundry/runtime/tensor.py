"""``ShardTensor`` — an owning full tensor beside the global type that shards it.

``Module.prepare`` writes one canonical tensor per weight and the HIR keeps
the global ``TensorType``, so what a checkpoint hands back is every program's
data at once. ``local`` is the one place that narrows it to the program a
``Placement`` names.

See [runtime §1.7](docs/spec/runtime.md#17-shardtensor).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tilefoundry.ir.types.shard import (
    Placement,
    Topology,
    flatten,
    level_axes,
    positions_at,
    shard_layout_of,
)
from tilefoundry.ir.types.shard.layout_algebra import idx2crd
from tilefoundry.ir.types.shard.shard_layout import split_target_axes
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.ir.types.utils import local_type_of


def _one_of(tensor: torch.Tensor, axis: int, count: int, position: int) -> torch.Tensor:
    """The *position*-th of *count* equal parts of *tensor* along *axis*, as a view."""
    extent = tensor.shape[axis]
    if extent % count:
        raise ValueError(
            f"ShardTensor: axis {axis} has extent {extent}, which its mesh axis of "
            f"{count} positions does not divide; a shard would not be one slice"
        )
    step = extent // count
    return tensor.narrow(axis, position * step, step)


@dataclass(frozen=True)
class ShardTensor:
    """An owning full tensor plus the global ``TensorType`` that shards it."""

    tensor: torch.Tensor
    type: TensorType

    def local(
        self, topologies: tuple[Topology, ...], placement: "Placement | None"
    ) -> torch.Tensor:
        """The slice this invocation's program ids select out of ``tensor``.

        A weight with no distribution is every program's whole tensor, and so
        is an axis the mesh only broadcasts over. Each ``Split`` narrows its
        tensor axis to the one part its level's program id lands on, and a
        level the placement leaves as ``None`` is left whole for the device to
        divide.
        """
        shard = shard_layout_of(self.type.layout)
        if shard is None:
            return self.tensor
        levels = tuple(topology.name for topology in shard.mesh.topologies)
        if placement is None:
            raise ValueError(
                f"ShardTensor: {tuple(self.type.shape)} is sharded over {levels}, and "
                f"no Placement says which program this is; pass placement= to load()"
            )
        ids = placement.program_ids(topologies)
        at = {topology.name: index for index, topology in enumerate(topologies)}
        mesh_shape = flatten(shard.mesh.layout.shape)
        tensor_axes = split_target_axes(shard, self.type.shape)
        out = self.tensor
        for topology, mesh_axes in zip(shard.mesh.topologies, level_axes(shard.mesh)):
            if topology.name not in at:
                declared = tuple(one.name for one in topologies) or ("none",)
                raise ValueError(
                    f"ShardTensor: the mesh shards over level {topology.name!r}, which "
                    f"the declared topology levels {declared} do not name"
                )
            program_id = ids[at[topology.name]]
            if program_id is None:
                continue
            coord = idx2crd(program_id, *positions_at(shard.mesh, topology.name))
            for mesh_axis, position in zip(mesh_axes, coord, strict=True):
                tensor_axis = tensor_axes[mesh_axis]
                if tensor_axis is None:
                    continue
                out = _one_of(out, tensor_axis, mesh_shape[mesh_axis], position)
        self._check(out, ids, at)
        return out

    def _check(self, out: torch.Tensor, ids: tuple, at: dict) -> None:
        """Hold the narrowed shape against what ``local_type_of`` says one shard is.

        The comparison only applies once every level of the mesh has an id,
        because a partly-projected shape is not something the type side names.
        """
        shard = shard_layout_of(self.type.layout)
        if any(ids[at[topology.name]] is None for topology in shard.mesh.topologies):
            return
        want = tuple(local_type_of(self.type).shape)
        if tuple(out.shape) != want:
            raise ValueError(
                f"ShardTensor: program ids {ids} select shape {tuple(out.shape)} out of "
                f"{tuple(self.type.shape)}, while one shard of it is {want}"
            )


__all__ = ["ShardTensor"]
