"""``ShardTensor`` — an owning full tensor beside the global type that shards it.

``Module.prepare`` writes one canonical tensor per weight and the HIR keeps
the global ``TensorType``, so what a checkpoint hands back is every program's
data at once. ``to_local`` is the one place that narrows it to the program a
``Placement`` names, through the same algebra the device side runs.

See [runtime §1.10](docs/spec/runtime.md#110-runtimetensorpy).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tilefoundry.ir.types.shard import (
    Placement,
    Topology,
    shard_layout_of,
)
from tilefoundry.ir.types.shard.local import local_layout_and_offset
from tilefoundry.ir.types.tensor_type import TensorType


@dataclass(frozen=True)
class ShardTensor:
    """An owning full tensor plus the global ``TensorType`` that shards it."""

    tensor: torch.Tensor
    type: TensorType

    def to_local(
        self, topologies: tuple[Topology, ...], placement: "Placement | None"
    ) -> torch.Tensor:
        """The slice this invocation's program ids select out of ``tensor``.

        The same algebra the device side runs, asked with a ``Placement``
        instead of a hardware id: ``local_layout_and_offset`` says what this
        instance holds and where it begins, and one strided view takes it. A
        weight with no distribution is every program's whole tensor, and so is
        a level the placement leaves unfixed -- ``cta`` and ``thread`` are the
        device's to divide.
        """
        shard = shard_layout_of(self.type.layout)
        if shard is None:
            return self.tensor
        topology_levels = tuple(topology.name for topology in shard.mesh.topologies)
        if placement is None:
            raise ValueError(
                f"ShardTensor: {tuple(self.type.shape)} is sharded over {topology_levels}, and "
                f"no Placement says which program this is; pass placement= to load()"
            )
        declared = tuple(topology.name for topology in topologies)
        missing = tuple(name for name in topology_levels if name not in declared)
        if missing:
            raise ValueError(
                f"ShardTensor: the mesh shards over level {missing[0]!r}, which the "
                f"declared topology levels {declared or ('none',)} do not name"
            )
        at = {name: index for index, name in enumerate(declared)}
        ids = placement.program_ids(topologies)
        layout, offset = local_layout_and_offset(
            shard, tuple(self.type.shape), tuple(ids[at[name]] for name in topology_levels)
        )
        if tuple(self.tensor.stride()) != tuple(layout.strides):
            raise ValueError(
                f"ShardTensor: the tensor is laid out {tuple(self.tensor.stride())} "
                f"while its type says {tuple(layout.strides)}; the offset is counted "
                f"in the type's strides, so the tensor must be laid out in them"
            )
        return self.tensor.as_strided(
            layout.shape, layout.strides, storage_offset=self.tensor.storage_offset() + offset
        )


__all__ = ["ShardTensor"]
