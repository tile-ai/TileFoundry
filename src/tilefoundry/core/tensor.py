"""``ShardTensor`` -- one global type and its data, packed or scattered.

Packed data is one global logical tensor; a ``Partial`` value holds its reduced
value, which is exact because type inference admits a ``Partial`` only into
consumers that commute with its reduction. Scattered data is one piece per
``Placement``, for a value that differs from unit to unit. Each piece has
``local_type(type)``.

See [runtime §1.10](docs/spec/runtime.md#110-shardtensor).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.types import Placement
from tilefoundry.ir.types.shard_layout import (
    Partial,
    _local_readback,
    local_layout,
    logical_layout_and_offset,
    program_coordinates,
    shard_layout_of,
)
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.utils.lazy import torch

_COMBINE = {"sum": "sum", "max": "amax", "min": "amin"}


def local_type(type: TensorType) -> TensorType:
    """The type of the piece one unit holds: the Local projection of *type*."""
    shard = shard_layout_of(getattr(type, "layout", None))
    if shard is None:
        return type
    layout = _local_readback(local_layout(shard, type.shape, (0,) * len(shard.attrs))[0])
    return replace(type, shape=layout.shape, layout=layout)


@dataclass(frozen=True)
class ShardTensor:
    """A global ``TensorType`` and its data, packed or one piece per ``Placement``."""

    type: TensorType
    tensor: "torch.Tensor | tuple[tuple[Placement, torch.Tensor], ...]"

    @property
    def packed(self) -> bool:
        """Whether the data is one global logical tensor."""
        return not isinstance(self.tensor, tuple)

    @property
    def placements(self) -> tuple[Placement, ...]:
        """The units a scattered tensor holds a piece for; empty when packed."""
        return () if self.packed else tuple(placement for placement, _ in self.tensor)

    def local(self, placement: "Placement | None") -> "torch.Tensor":
        """The data *placement* holds; the whole tensor when nothing divides it."""
        if not self.packed:
            for held, data in self.tensor:
                if held == placement:
                    return data
            raise ValueError(f"ShardTensor: no piece is held by {placement}")
        shard = shard_layout_of(self.type.layout)
        if shard is None:
            return self.tensor
        if placement is None:
            levels = tuple(getattr(topology, "name", topology) for topology in shard.mesh.topologies)
            raise ValueError(
                f"ShardTensor: {tuple(self.type.shape)} is sharded over {levels}, and "
                f"no Placement says which program this is; pass placement= to load()"
            )
        layout, offset = logical_layout_and_offset(
            shard, tuple(self.type.shape), placement.program_ids(shard.mesh.topologies)
        )
        base = self.tensor.contiguous()
        return base.as_strided(layout.shape, layout.strides, base.storage_offset() + offset)

    def scatter(self, placements: tuple[Placement, ...]) -> "ShardTensor":
        """One piece per placement; a scattered tensor is returned as it is."""
        if not self.packed:
            return self
        return ShardTensor(self.type, tuple((p, self.local(p)) for p in placements))

    def gather(self) -> "ShardTensor":
        """One global tensor, by the meaning of ``type``.

        ``Split`` pieces are placed side by side, ``Broadcast`` pieces are one
        value so any of them serves, and ``Partial`` pieces combine by their
        reduction.
        """
        if self.packed:
            return self
        shard = shard_layout_of(self.type.layout)
        if shard is None:
            raise ValueError(
                "ShardTensor: a scattered value whose type has no ShardLayout has no "
                "global value to gather"
            )
        partial = [axis for axis, attr in enumerate(shard.attrs) if isinstance(attr, Partial)]
        reductions = {shard.attrs[axis].reduction for axis in partial}
        if len(reductions) > 1 or reductions - _COMBINE.keys():
            raise ValueError(f"ShardTensor: cannot combine Partial reductions {sorted(reductions)}")
        shape = tuple(self.type.shape)
        first = self.tensor[0][1]
        parts: dict[tuple, torch.Tensor] = {}
        for placement, data in self.tensor:
            ids = placement.program_ids(shard.mesh.topologies)
            coordinates = program_coordinates(shard.mesh, ids)
            key = tuple(coordinates[axis] for axis in partial)
            if key not in parts:
                parts[key] = torch.empty(shape, dtype=first.dtype, device=first.device)
            layout, offset = logical_layout_and_offset(shard, shape, ids)
            parts[key].as_strided(layout.shape, layout.strides, offset).copy_(data)
        values = list(parts.values())
        if len(values) == 1:
            return ShardTensor(self.type, values[0])
        combine = getattr(torch.stack(values), _COMBINE[reductions.pop()])
        return ShardTensor(self.type, combine(dim=0))

    def reshard(self, type: TensorType) -> "ShardTensor":
        """The same value under *type*: packed data only changes type.

        A scattered value is gathered first, unless neither type has a
        ShardLayout: then each unit only moves its own value, which stays its own.
        """
        if not self.packed and shard_layout_of(self.type.layout) is None and (
            shard_layout_of(type.layout) is None
        ):
            return ShardTensor(type, self.tensor)
        return ShardTensor(type, self.gather().tensor)


__all__ = ["ShardTensor", "local_type"]
