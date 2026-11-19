"""Which topology owns each storage level of a target.

``SHARED_OWNER`` names storage that every unit of the target shares.

See [target §11](docs/spec/target.md#11-target-facts-projection).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from tilefoundry.ir import expr

__all__ = ["SHARED_OWNER", "StorageOwners"]

SHARED_OWNER = "target"


@dataclass(frozen=True)
class StorageOwners:
    """Storage owners and the target's topology order, coarse to fine.

    The empty value, ``StorageOwners()``, stands for no target: no storage is
    private to a unit.
    """

    order: tuple[str, ...] = ()
    owner: Mapping[expr.StorageKind, str] = field(default_factory=lambda: MappingProxyType({}))

    def private(self, storage, topology) -> bool:
        """Whether *storage* is private to one unit of *topology* under these owners."""
        storage = expr.resolve_storage(storage)
        storage = expr.StorageKind.RMEM if storage is expr.StorageKind.UMAT else storage
        owner = self.owner.get(storage, SHARED_OWNER)
        return (
            owner != SHARED_OWNER
            and topology in self.order
            and self.order.index(topology) <= self.order.index(owner)
        )
