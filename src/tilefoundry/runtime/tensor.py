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
from tilefoundry.ir.types.shard.local import local_window
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
        instead of a hardware id: ``local_window`` says which part of each
        tensor axis is this instance's, and one basic slice takes it. A weight
        with no distribution is every program's whole tensor, and so is a level
        the placement leaves unfixed -- ``cta`` and ``thread`` are the device's
        to divide.
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
        declared = tuple(topology.name for topology in topologies)
        missing = tuple(name for name in levels if name not in declared)
        if missing:
            raise ValueError(
                f"ShardTensor: the mesh shards over level {missing[0]!r}, which the "
                f"declared topology levels {declared or ('none',)} do not name"
            )
        at = {name: index for index, name in enumerate(declared)}
        ids = placement.program_ids(topologies)
        window = local_window(
            shard, tuple(self.type.shape), tuple(ids[at[name]] for name in levels)
        )
        return self.tensor[
            tuple(slice(start, start + held) for start, held in window)
        ]


__all__ = ["ShardTensor"]
