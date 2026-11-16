"""``Placement`` — which unit a host invocation is speaking for.

A checkpoint states one global tensor per weight and a ``ShardLayout`` states
how the mesh divides it. Neither says which of those units the process
calling ``load`` occupies, because that is not a property of the model. A
``Placement`` answers exactly that one question, and nothing else: it is read
on the host and never reaches a ``TensorType``, a checkpoint or a kernel
argument.

See [shard §5](docs/spec/shard.md#5-mesh).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from .layout import apply, plain_layout, size
from .mesh import Mesh, offset, sub_meshes
from .shard_layout import mesh_coords
from .topology import Topology, topology_name

__all__ = ["Placement"]


@dataclass(frozen=True)
class Placement:
    """Where this invocation runs: a unit id for each topology it names.

    Data, not a callable: which unit a loaded module speaks for is fixed
    when it is loaded, so a placement that answered differently on a later
    call would let one module change identity while holding weights narrowed
    for the old one. A level nobody named is left unfixed -- ``cta`` and
    ``thread`` are chosen by the device, and a host that guessed ``0`` for
    them would hand back one thread's data as if it were the card's.
    """

    ids: Mapping[str, int]

    def unit_ids(self, topologies: "tuple[Topology | str, ...]") -> tuple[int | None, ...]:
        """This invocation's unit id at each of *topologies*, in that order."""
        return tuple(self.ids.get(topology_name(topology)) for topology in topologies)

    def coords(self, mesh: Mesh) -> tuple[int | None, ...]:
        """This placement's coord on each mesh axis; None where its topology is not fixed."""
        return mesh_coords(mesh, self.unit_ids(mesh.topologies))

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> "tuple[Placement, ...]":
        """Every placement of *mesh*: each unit id is its topology's layout value plus offset."""
        parts = sub_meshes(mesh)
        return tuple(
            cls(
                {
                    topology_name(part.topologies[0]): apply(plain_layout(part.layout), index)
                    + offset(part)
                    for part, index in zip(parts, at, strict=True)
                }
            )
            for at in product(*(range(size(part.layout)) for part in parts))
        )
