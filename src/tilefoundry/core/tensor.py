"""``ShardTensor`` -- one global type and its data, in a global or a local view.

A global view is one global logical tensor; a ``Partial`` value holds its
reduced value, which is exact because type inference admits a ``Partial`` only
into consumers that commute with its reduction. A local view is one piece per
``Placement``, for a value that differs from placement to placement. Each piece
has ``local_type(type)``. The two names are OneFlow's.

See [runtime §1.10](docs/spec/runtime.md#110-shardtensor).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from tilefoundry.ir.types import Placement
from tilefoundry.ir.types.layout import ComposedLayout, Layout
from tilefoundry.ir.types.mesh import _named
from tilefoundry.ir.types.shard_layout import (
    ShardLayout,
    _local_readback,
    local_layout,
    local_layout_and_offset,
    mesh_coords,
    partial_reductions_by_axis,
    shard_layout_of,
)
from tilefoundry.ir.types.tensor_type import TensorType
from tilefoundry.utils.lazy import torch

_COMBINE = {"sum": "sum", "max": "amax", "min": "amin"}


def local_type(type: TensorType) -> TensorType:
    """The type of the piece one placement holds: the Local projection of *type*."""
    shard = shard_layout_of(getattr(type, "layout", None))
    if shard is None:
        return type
    layout = _local_readback(local_layout(shard, type.shape, (0,) * len(shard.attrs))[0])
    return replace(type, shape=layout.shape, layout=layout)


@dataclass(frozen=True)
class ShardTensor:
    """A global ``TensorType`` and its data, in a global view or one piece per ``Placement``."""

    type: TensorType
    tensor: "torch.Tensor | tuple[tuple[Placement, torch.Tensor], ...]"

    @property
    def is_global(self) -> bool:
        """Whether the data is one global logical tensor."""
        return not isinstance(self.tensor, tuple)

    @property
    def placements(self) -> tuple[Placement, ...]:
        """The placements a local view holds a piece for; empty in a global view."""
        return () if self.is_global else tuple(placement for placement, _ in self.tensor)

    def to_local(self, placement: "Placement | None") -> "torch.Tensor":
        """The data *placement* holds; the whole tensor when nothing divides it.

        The host side of the CUDA runtime's ``local_tensor``.
        """
        if not self.is_global:
            for held, data in self.tensor:
                if held == placement:
                    return data
            raise ValueError(f"ShardTensor: no piece is held by {placement}")
        shard = shard_layout_of(self.type.layout)
        if shard is None:
            return self.tensor
        if placement is None:
            raise ValueError(
                f"ShardTensor: {tuple(self.type.shape)} is sharded over {_named(shard.mesh)}, and "
                f"no Placement says which program this is; pass placement= to load()"
            )
        if any(partial_reductions_by_axis(shard)):
            raise ValueError(
                "ShardTensor: a global Partial value holds its reduced value; "
                "no placement's own part exists"
            )
        return self._window(self.tensor.contiguous(), shard, placement)

    def to_global(self) -> "ShardTensor":
        """One global tensor, by the meaning of ``type``.

        ``Split`` pieces are placed side by side, ``Broadcast`` pieces are one
        value so any of them serves, and ``Partial`` pieces combine by their
        reduction.
        """
        if self.is_global:
            return self
        shard = shard_layout_of(self.type.layout)
        if shard is None:
            raise ValueError(
                "ShardTensor: a local view whose type has no ShardLayout has no "
                "global value to assemble"
            )
        by_axis = partial_reductions_by_axis(shard)
        partial = [axis for axis, kind in enumerate(by_axis) if kind is not None]
        reductions = {kind for kind in by_axis if kind is not None}
        if len(reductions) > 1 or reductions - _COMBINE.keys():
            raise ValueError(f"ShardTensor: cannot combine Partial reductions {sorted(reductions)}")
        first = self.tensor[0][1]
        parts: dict[tuple, torch.Tensor] = {}
        for placement, data in self.tensor:
            coord = mesh_coords(shard.mesh, placement.program_ids(shard.mesh.topologies))
            key = tuple(coord[axis] for axis in partial)
            if key not in parts:
                parts[key] = torch.empty(
                    tuple(self.type.shape), dtype=first.dtype, device=first.device
                )
            self._window(parts[key], shard, placement).copy_(data)
        values = list(parts.values())
        if len(values) == 1:
            return ShardTensor(self.type, values[0])
        combine = getattr(torch.stack(values), _COMBINE[reductions.pop()])
        return ShardTensor(self.type, combine(dim=0))

    def reshard(self, type: TensorType) -> "ShardTensor":
        """The same value under *type*: a global view only changes type.

        A local view is assembled first, unless neither type has a ShardLayout:
        then each placement only moves its own value, which stays its own.
        """
        if not self.is_global and shard_layout_of(self.type.layout) is None and (
            shard_layout_of(type.layout) is None
        ):
            return ShardTensor(type, self.tensor)
        return ShardTensor(type, self.to_global().tensor)

    def _window(self, whole, shard: ShardLayout, placement: Placement):
        """The part of row-major *whole* that *placement* holds.

        Host data is the logical tensor, so the cut keeps the shard's modes and
        reads them with row-major strides; the address strides may not say where
        a placement's part is, since a private Split has stride zero.
        """
        address = shard.layout.outer if isinstance(shard.layout, ComposedLayout) else shard.layout
        host = replace(shard, layout=Layout(address.shape))
        coord = mesh_coords(shard.mesh, placement.program_ids(shard.mesh.topologies))
        layout, offset = local_layout_and_offset(host, tuple(self.type.shape), coord)
        return whole.as_strided(layout.shape, layout.strides, whole.storage_offset() + offset)


def local_placements(*tensors: ShardTensor) -> tuple[Placement, ...]:
    """The placements the local views among *tensors* hold; empty when every one is global."""
    held = [tensor.placements for tensor in tensors if not tensor.is_global]
    if any(placements != held[0] for placements in held[1:]):
        raise ValueError("ShardTensor: local-view inputs are held by different placements")
    return held[0] if held else ()


__all__ = ["ShardTensor", "local_placements", "local_type"]
