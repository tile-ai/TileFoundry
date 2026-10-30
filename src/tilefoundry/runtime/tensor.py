"""``ShardTensor`` — an owning full tensor beside the global type that shards it.

``Module.prepare`` writes one canonical tensor per weight and the HIR keeps
the global ``TensorType``, so what a checkpoint hands back is every program's
data at once. ``to_local`` is the one place that narrows it to the program a
``Placement`` names, through the same algebra the device side runs.

See [runtime §1.10](docs/spec/runtime.md#110-runtimetensorpy).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.types import Placement, Topology
from tilefoundry.ir.types.layout import Layout, normalize_layout
from tilefoundry.ir.types.shard_layout import (
    _local_readback,
    local_layout,
    local_layout_and_offset,
    shard_layout_of,
)
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.utils.lazy import torch


def _strided_local_view(tensor, layout, offset):
    """Apply the shared local projection to an already matching tensor engine."""
    return tensor.as_strided(
        layout.shape, layout.strides, storage_offset=tensor.storage_offset() + offset
    )


def logical_local_view(tensor, held: TensorType, coordinates: tuple[int | None, ...]):
    """Select logical elements through the canonical Split/domain-axis algebra.

    Reference inputs are full logical tensors, even when the HIR declares
    independent device engines with zero physical strides. Construct the
    canonical dense address map over the existing layout modes, then use the
    same projection as checkpoint loading. Physical offsets and swizzles do
    not alter which logical elements a participant owns.
    """
    shard = shard_layout_of(held.layout)
    if shard is None:
        return tensor
    canonical = replace(shard, layout=Layout(shard.layout.shape))
    layout, offset = local_layout(canonical, tuple(held.shape), coordinates)
    return _strided_local_view(tensor.contiguous(), _local_readback(layout), offset)


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
        written = normalize_layout(layout).strides
        actual = normalize_layout(
            Layout(tuple(self.tensor.shape), tuple(self.tensor.stride()))
        ).strides
        if actual != written:
            raise ValueError(
                f"ShardTensor: the tensor is laid out {tuple(self.tensor.stride())} "
                f"while its type says {tuple(layout.strides)}; the offset is counted "
                f"in the type's strides, so the tensor must be laid out in them"
            )
        return _strided_local_view(self.tensor, layout, offset)


__all__ = ["ShardTensor"]
