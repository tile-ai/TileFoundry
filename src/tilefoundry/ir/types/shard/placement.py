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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from tilefoundry.ir.types.shard.mesh import Topology


@dataclass(frozen=True)
class Placement:
    """Host-side execution context: which program this invocation is.

    The stored callable is asked for one program id per ordered ``Topology``,
    so a caller reads its answer positionally rather than by name. ``None``
    stands for a level the host has not fixed -- ``cta`` and ``thread`` are
    chosen by the device, and a host that guessed ``0`` for them would hand
    back one thread's data as if it were the card's.
    """

    program_ids_getter: Callable[[tuple[Topology, ...]], Sequence[int | None]]

    def program_ids(self, topologies: tuple[Topology, ...]) -> tuple[int | None, ...]:
        """This invocation's program id at each of *topologies*, in that order."""
        ids = tuple(self.program_ids_getter(tuple(topologies)))
        if len(ids) != len(topologies):
            raise ValueError(
                f"Placement: {len(ids)} program ids for {len(topologies)} topology "
                f"levels {tuple(topology.name for topology in topologies)}; one id "
                f"is required per Topology"
            )
        return ids


__all__ = ["Placement"]
