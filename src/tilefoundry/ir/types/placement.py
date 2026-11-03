"""``Placement`` — which program a host invocation is speaking for.

A checkpoint states one global tensor per weight and a ``ShardLayout`` states
how the mesh divides it. Neither says which of those positions the process
calling ``load`` occupies, because that is not a property of the model. A
``Placement`` answers exactly that one question, and nothing else: it is read
on the host and never reaches a ``TensorType``, a checkpoint or a kernel
argument.

See [shard §5](docs/spec/shard.md#5-mesh).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from tilefoundry.ir.types.layout import flatten
from tilefoundry.ir.types.mesh import Mesh, Topology, levels
from tilefoundry.ir.types.stride import crd2idx


@dataclass(frozen=True)
class Placement:
    """Host-side execution context: which program this invocation is.

    Data, not a callable: which program a loaded module speaks for is fixed
    when it is loaded, so a placement that answered differently on a later
    call would let one module change identity while holding weights narrowed
    for the old one. A level nobody named is left unfixed -- ``cta`` and
    ``thread`` are chosen by the device, and a host that guessed ``0`` for
    them would hand back one thread's data as if it were the card's.
    """

    ids: Mapping[str, int]

    def program_ids(self, topologies: "tuple[Topology | str, ...]") -> tuple[int | None, ...]:
        """This invocation's program id at each of *topologies*, in that order."""
        return tuple(self.ids.get(getattr(topology, "name", topology)) for topology in topologies)

    @classmethod
    def from_env(cls, topology_level: str = "gpu") -> "Placement":
        """The peer this process is, out of the ones a launcher started.

        ``torchrun``, ``deepspeed`` and ``mpirun`` all put the local rank in
        the environment before the script runs, which is the one moment the
        answer is settled.
        """
        rank = os.environ.get("LOCAL_RANK")
        if rank is None:
            raise ValueError(
                "Placement.from_env: no LOCAL_RANK in the environment; a launcher "
                "sets it, so pass Placement({topology_level!r}: rank) yourself instead"
            )
        return cls({topology_level: int(rank)})


def units(mesh: Mesh) -> tuple[tuple[tuple[int, ...], Placement], ...]:
    """Every unit of *mesh*: its mesh coordinates, row-major, and the Placement naming it.

    A level's program id is that level's coordinates read through the level's
    arrangement, the inverse of what a local view does with the id.
    """
    arrangements = tuple(flatten(arrangement) for arrangement in levels(mesh))
    names = tuple(getattr(topology, "name", topology) for topology in mesh.topologies)
    found = []
    for coordinates in product(*(range(extent) for extent in flatten(mesh.layout).shape)):
        ids, begin = {}, 0
        for name, flat in zip(names, arrangements, strict=True):
            count = len(flat.shape)
            ids[name] = crd2idx(coordinates[begin : begin + count], flat.shape, flat.strides)
            begin += count
        found.append((tuple(coordinates), Placement(ids)))
    return tuple(found)


__all__ = ["Placement", "units"]
